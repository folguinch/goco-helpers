"""Custom JSON encoders and hooks."""
from typing import Dict
from pathlib import Path
import json

import astropy.units as u

class CustomObjEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'unit'):
            return {'__quantity__': True,
                    'val': obj.value,
                    'unit': f'{obj.unit}'}
        elif hasattr(obj, 'stem'):
            return {'__path__': True, 'val': f'{obj}'}
        return super().default(obj)

def custom_hooks(dct):
    if '__quantity__' in dct:
        return dct['val'] * u.Unit(dct['unit'])
    elif '__path__' in dct:
        return Path(dct['val'])
    return dct
