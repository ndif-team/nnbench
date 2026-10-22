"""Prototype for nnsight 0.8.x: pickle each Envoy as a reference (`Envoy:<path>`) to the receiver's
own envoy at the same path, instead of shipping the whole envoy tree with every invoke.

Loaded at interpreter start in every process that has this directory on PYTHONPATH -- the client
and vLLM's spawned EngineCore alike -- and active only when NNSIGHT_ENVOY_BYREF=1. Both halves are
needed: the client writes the reference, the engine resolves it through its persistent-object map
(built once at model load). Not in any image: the `*/byref` configs and slurm/diag_cost.py put
this directory on PYTHONPATH. A prototype of a fix for nnsight, measured, not shipped.
"""
import os

if os.environ.get("NNSIGHT_ENVOY_BYREF") == "1":
    try:
        from nnsight.intervention.envoy import Envoy
        from nnsight.intervention.serialization import CustomCloudPickler
        from nnsight.modeling.mixins.remotable import Remotable
    except ImportError:                 # an interpreter without nnsight (a build step, a tool)
        pass
    else:
        _persistent_id = CustomCloudPickler.persistent_id

        def persistent_id(self, obj):
            if isinstance(obj, Envoy):
                return f"Envoy:{obj.path}"
            return _persistent_id(self, obj)

        _objects = Remotable._remoteable_persistent_objects

        def _remoteable_persistent_objects(self):
            objects = _objects(self)
            for envoy in self.modules():
                objects[f"Envoy:{envoy.path}"] = envoy
            return objects

        CustomCloudPickler.persistent_id = persistent_id
        Remotable._remoteable_persistent_objects = _remoteable_persistent_objects
