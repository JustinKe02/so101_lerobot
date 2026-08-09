#!/usr/bin/env python

# Copyright 2026 Dexmal
#
# Licensed under the MIT License. See LICENSE in this directory.

"""Vendored realtime-vla-v2 PI0.5 Triton kernels.

Import the concrete kernel modules only from the lazy runtime loader. Keeping
this package initializer empty allows LeRobot and PI0.5 to import on machines
without Triton or CUDA.
"""
