"""Custom JSON encoders and hooks."""
from typing import Dict
from pathlib import Path
import json

import astropy.units as u

class CustomObjEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'unit'):
            val = obj.value
            if hasattr(val, 'size'):
                val = list(map(float, val))
            return {'__quantity__': True,
                    'val': val,
                    'unit': f'{obj.unit}'}
        elif hasattr(obj, 'stem'):
            return {'__path__': True, 'val': f'{obj}'}

        return super().default(obj)

def custom_hooks(dct):
    if '__quantity__' in dct:
        return dct['val'] * u.Unit(dct['unit'])
    elif '__quantity_list__' in dct:
        vals = dct['val']
        units = dct['unit']
        return [val * u.Unit(unit) for val, unit in zip(vals, units)]
    elif '__path__' in dct:
        return Path(dct['val'])
    return dct
