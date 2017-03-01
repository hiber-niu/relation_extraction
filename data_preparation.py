#!/usr/bin/env python
# -*- coding: utf-8 -*-
###########################################
#  Prepare data for relation extraction.  #
#                                         #
#                 date:2016-12-06         #
#             author: hiber.niu@gmail.com #
###########################################
import sys
sys.path.append("./packages")
import data_preparation_tool as dpt

import datetime
from datetime import date


today = date.today()
prefix = "%s_%s_%s"%(today.year, today.month, today.day)

output_dir = r"./output"
positive_samples_path = r"./input/positive.txt"
negative_samples_path = r"./input/negative.txt"

start_time = datetime.datetime.now()

# this pipeline will produce files with all possible outputs
dpt.run_data_preparation_pipeline(positive_samples_path, negative_samples_path, prefix, output_dir)

run_time = datetime.datetime.now() - start_time
print("Finished producing files, took:%s"%run_time)
