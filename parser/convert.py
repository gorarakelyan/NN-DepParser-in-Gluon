import os

import argparse

from config import *

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', default='', type=str, help='Data Path')
parser.add_argument('--output_file', default='', type=str, help='Output File Path')
args = parser.parse_args()

cmd = 'java -jar ' + PATH_PENN2CONLL + ' -stopOnError=False -raw < {i} > {o}'.format(i=args.input_file, o=args.output_file)
os.system(cmd)