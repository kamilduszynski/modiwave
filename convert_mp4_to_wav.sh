#!/bin/bash

INPUT_FILE=$1
OUTPUT_FILE=$2

ffmpeg -i $INPUT_FILE -ac 2 -f wav $OUTPUT_FILE