# -*- coding: utf-8 -*-
# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Description: Some function for configuration of the models.
"""
import yaml
from argparse import Namespace


def parse_args(parser):
    """Parse the arguments.
    Args:
        parser: An instance of argparse.ArgumentParser.
    Returns:
        args: An instance of argparse.Namespace.
    """
    args = parser.parse_args()
    if not args.conf_path:
        raise ValueError(f"--conf_path can't be None.")
    args_dict = yaml.load(open(args.conf_path).read())
    args = vars(args)
    for k, v in args.items():
        if v:
            args_dict[k] = v
    args = dict2namespace(args_dict)
    return args


def dict2namespace(dic):
    """Convert a `dict` to an `argparse.Namespace`.
    Args:
        dic: A `dict`.
    Returns:
        args: An instance of argparse.Namespace.
    """
    for k, v in dic.items():
        if isinstance(v, dict):
            v = dict2namespace(v)
            dic[k] = v
    return Namespace(**dic)

