"""Offline Earth Engine harness.

Earth Engine needs a server to look up its algorithms, so code that builds EE
expressions cannot normally run without credentials. This harness registers
permissive stand-ins for every algorithm the client library calls, intercepts
the network calls (getInfo, map ids, thumbnails) and lets a test answer them.

It catches client-side mistakes - wrong method names, wrong argument counts
or keywords, Python errors inside mapped functions, values that cannot be
serialized - and lets the post-processing run on realistic responses. It does
not check server-side semantics such as band names; scripts/ee_smoke_test.py
does that against the real service.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import re
from types import SimpleNamespace
from typing import Any, Callable

import ee
from ee import apifunction, data, deprecation, serializer
from ee.encodable import Encodable

_CLASSES = [
    ee.Image, ee.ImageCollection, ee.Collection, ee.FeatureCollection, ee.Feature,
    ee.Element, ee.Geometry, ee.Reducer, ee.Filter, ee.Date, ee.List, ee.Dictionary,
    ee.Number, ee.String, ee.Terrain, ee.Projection, ee.Kernel, ee.Array, ee.DateRange,
    ee.Join, ee.PixelType, ee.ErrorMargin,
]
_CONCRETE = {"Image", "ImageCollection", "Feature", "FeatureCollection", "Geometry", "Reducer",
             "Filter", "Date", "List", "Dictionary", "Number", "String", "Projection", "Kernel",
             "Array", "DateRange", "Element", "PixelType", "ErrorMargin", "Join"}

_PATTERNS = [
    (re.compile(r"ApiFunction\.(?:call_|apply_|lookup)\(\s*['\"]([\w.]+)['\"]"), None),
    (re.compile(r"ApiFunction\.(?:call_|apply_|lookup)\(\s*self\.name\(\)\s*\+\s*['\"](\.\w+)['\"]"), "self"),
    (re.compile(r"ApiFunction\.(?:call_|apply_|lookup)\(\s*cls\.name\(\)\s*\+\s*['\"](\.\w+)['\"]"), "self"),
    (re.compile(r"ApiFunction\.(?:call_|apply_|lookup)\(\s*REDUCE_PREFIX\s*\+\s*['\"](\.\w+)['\"]"), "reduce"),
]


class LooseFunction(apifunction.ApiFunction):
    """Accepts any positional or keyword arguments; returns a typed object."""

    def nameArgs(self, args, extra_keyword_args=None):  # noqa: N802 - EE naming
        named = {f"_a{i}": v for i, v in enumerate(args)}
        named.update(extra_keyword_args or {})
        return named

    def promoteArgs(self, args):  # noqa: N802
        out = {}
        for k, v in args.items():
            if callable(v) and not isinstance(v, Encodable):
                # Python callbacks (map, iterate) become EE functions; this runs them
                v = ee._Promote(v, "Algorithm")  # noqa: SLF001
            out[k] = v
        return out


def _return_type(annotation: Any) -> str:
    a = str(annotation).strip("'\"")
    a = a.split("|")[0].strip().split("[")[0]
    a = a.split(".")[-1]
    return a if a in _CONCRETE else "Object"


def _registry() -> dict[str, LooseFunction]:
    reg: dict[str, LooseFunction] = {}

    def add(name: str, ret: str, owner: str):
        if name not in reg or reg[name].getSignature()["returns"] == "Object":
            reg[name] = LooseFunction(name, {
                "args": [{"name": "_a0", "type": owner, "optional": True}],
                "returns": ret,
            })

    for cls in _CLASSES:
        owner = cls.__name__
        for attr, fn in vars(cls).items():
            if isinstance(fn, (staticmethod, classmethod)):
                fn = fn.__func__
            if not callable(fn):
                continue
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            ret = _return_type(inspect.signature(fn).return_annotation) \
                if inspect.signature(fn).return_annotation is not inspect.Signature.empty else "Object"
            for pat, kind in _PATTERNS:
                for m in pat.finditer(src):
                    if kind == "self":
                        name = owner + m.group(1)
                    elif kind == "reduce":
                        name = "reduce" + m.group(1)
                    else:
                        name = m.group(1)
                    add(name, ret if attr != "__init__" else owner, owner)
    for extra, ret in {"If": "Object", "Image.load": "Image", "Image.constant": "Image",
                       "Collection.loadTable": "FeatureCollection", "ImageCollection.load": "ImageCollection",
                       "Feature": "Feature", "Collection": "FeatureCollection",
                       "ImageCollection.fromImages": "ImageCollection", "Image.neq": "Image",
                       "Collection.geometry": "Geometry"}.items():
        add(extra, ret, extra.split(".")[0])
    return reg


class _Lazy(dict):
    """Registry that invents a permissive function for any unknown name."""

    def get(self, name, default=None):
        if name not in self:
            owner = name.split(".")[0] if "." in name else "Object"
            self[name] = LooseFunction(name, {"args": [{"name": "_a0", "type": owner,
                                                        "optional": True}],
                                              "returns": "Object"})
        return super().get(name, default)


@contextlib.contextmanager
def dry_run(responder: Callable[[str, Any], Any], log: list | None = None):
    """Run Earth Engine code offline. responder(kind, payload) answers network calls.

    kind is "value" (payload = serialized JSON string), "mapid", "thumb" or "video".
    """
    saved = {
        "init": apifunction.ApiFunction.initialize,
        "data_init": data.initialize,
        "dep": deprecation.InitializeDeprecatedAssets,
        "compute": data.computeValue,
        "mapid": data.getMapId,
        "thumbid": data.getThumbId,
        "thumburl": data.makeThumbUrl,
        "name_args": apifunction.ApiFunction.__dict__.get("nameArgs"),
        "promote_args": apifunction.ApiFunction.__dict__.get("promoteArgs"),
    }
    registry = _Lazy(_registry())

    def fake_init(cls):
        cls._api = registry

    def compute(obj):
        payload = serializer.toJSON(obj)
        json.loads(payload)
        if log is not None:
            log.append(payload)
        return responder("value", payload)

    def get_map_id(params):
        serializer.toJSON(params["image"])
        return {"mapid": "dry", "token": "", "tile_fetcher": SimpleNamespace(
            url_format=responder("mapid", params))}

    def get_thumb_id(params, *args, **kwargs):
        img = params.get("image")
        if img is not None:
            serializer.toJSON(img)
        kind = "video" if (params.get("format") == "gif" or params.get("framesPerSecond")) else "thumb"
        return {"thumbid": responder(kind, params), "token": ""}

    try:
        apifunction.ApiFunction.reset()
        apifunction.ApiFunction.initialize = classmethod(fake_init)
        # Constructors such as ee.Projection build ApiFunction objects directly,
        # so the permissive argument handling goes on the base class too.
        apifunction.ApiFunction.nameArgs = LooseFunction.nameArgs
        apifunction.ApiFunction.promoteArgs = LooseFunction.promoteArgs
        data.initialize = lambda *a, **k: None
        deprecation.InitializeDeprecatedAssets = lambda: None
        data.computeValue = compute
        data.getMapId = get_map_id
        data.getThumbId = get_thumb_id
        data.makeThumbUrl = lambda thumb: f"https://dry.run/thumb/{thumb['thumbid']}"
        ee.Reset()
        ee.Initialize(credentials=None, project="dry-run-project")
        yield
    finally:
        apifunction.ApiFunction.initialize = saved["init"]
        for attr, key in (("nameArgs", "name_args"), ("promoteArgs", "promote_args")):
            if saved[key] is None:
                if attr in apifunction.ApiFunction.__dict__:
                    delattr(apifunction.ApiFunction, attr)
            else:
                setattr(apifunction.ApiFunction, attr, saved[key])
        data.initialize = saved["data_init"]
        deprecation.InitializeDeprecatedAssets = saved["dep"]
        data.computeValue = saved["compute"]
        data.getMapId = saved["mapid"]
        data.getThumbId = saved["thumbid"]
        data.makeThumbUrl = saved["thumburl"]
        try:
            ee.Reset()
        except Exception:  # noqa: BLE001
            pass
