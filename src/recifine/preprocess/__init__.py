from __future__ import annotations
from typing import Callable, Dict

from .finer import preprocess_finer_ka, preprocess_finer_trad
from .ar import preprocess_ar_ka, preprocess_ar_trad
from .gk import preprocess_gk_ka, preprocess_gk_trad
from .foodbase import preprocess_foodbase_ka, preprocess_foodbase_trad
from .tasteset1 import preprocess_tasteset1_ka, preprocess_tasteset1_trad
from .tasteset2 import preprocess_tasteset2_ka, preprocess_tasteset2_trad
from .englishflowgraph import preprocess_englishflowgraph_ka, preprocess_englishflowgraph_trad
from .recifinegold import preprocess_recifinegold_ka, preprocess_recifinegold_trad

PREPROCESSORS: Dict[str, Callable] = {}

PREPROCESSORS["finer_ka"] = preprocess_finer_ka
PREPROCESSORS["finer_trad"] = preprocess_finer_trad

PREPROCESSORS["ar_ka"] = preprocess_ar_ka
PREPROCESSORS["ar_trad"] = preprocess_ar_trad

PREPROCESSORS["gk_ka"] = preprocess_gk_ka
PREPROCESSORS["gk_trad"] = preprocess_gk_trad

PREPROCESSORS["foodbase_ka"] = preprocess_foodbase_ka
PREPROCESSORS["foodbase_trad"] = preprocess_foodbase_trad

PREPROCESSORS["tasteset1_ka"] = preprocess_tasteset1_ka
PREPROCESSORS["tasteset1_trad"] = preprocess_tasteset1_trad

PREPROCESSORS["tasteset2_ka"] = preprocess_tasteset2_ka
PREPROCESSORS["tasteset2_trad"] = preprocess_tasteset2_trad

PREPROCESSORS["englishflowgraph_ka"] = preprocess_englishflowgraph_ka
PREPROCESSORS["englishflowgraph_trad"] = preprocess_englishflowgraph_trad

PREPROCESSORS["recifinegold_ka"] = preprocess_recifinegold_ka
PREPROCESSORS["recifinegold_trad"] = preprocess_recifinegold_trad
