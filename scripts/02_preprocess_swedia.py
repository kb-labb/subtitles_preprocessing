import argparse
import pandas as pd
import simplejson
import csv
import os
from sub_preproc.utils.utils import SILENCE, pandas_to_dict, add_silence
from sub_preproc.utils.make_chunks import make_chunks_isof

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--file")

    return parser.parse_args()   
    

def main():
    args = get_args()
    csv_file = args.file
    path = "/".join(csv_file.split("/")[:-1])
    output_path = "/".join(csv_file.split("/")[:-2])

    colnames = ['sub_id', 'place_name', 'sound_file_id', 'speaker_id', 'sub_start', 'sub_end', 'text']
    df = pd.read_csv(csv_file, sep = '|', names=colnames, quoting=csv.QUOTE_NONE)
    df.set_index('sub_id', inplace=True)

    g = df.groupby('sound_file_id')
    audio_files = list(g.groups.keys())
    
    #List of files with low quality gold standard transcription
    exclude_files = ['far_om_1s_far_om_2s', 'oru_ym_1s', 'vim_ow_1s', 'pit_om_1s', 'arj_yw_2s', 'arj_ow_1s', 'vim_yw_3s', 'pit_ym_3s', 'nys_yw_3s', 'bar_om_1s', 'sko_ym_3s', 'arj_ym_3s', 'nys_om_1s', 'hal_ym_1s', 'vim_om_3s', 'pit_yw_3s', 'far_ym_3s', 'far_ow_1s_far_ow_2s', 'far_yw_1s', 'fao_om_1s']


    for name in audio_files:
        subs = g.get_group(name)

        #Make file name from csv-file match audio file name
        if ".gloss" in name:
            name = name.replace(".gloss", "")
        elif "_gloss" in name:
            name = name.replace("_gloss", "")

        if name in exclude_files:
            continue

        area = "/".join(subs['place_name'].iloc[0].split("/")[-2:])
        audio_path = path + "/audio/" + area + "/" + name + ".mp3"

        if os.path.exists(audio_path):
            subs_dict = pandas_to_dict(subs, audio_path, area)

            subs_with_silence = add_silence(subs_dict)

            #exclude entire sub containing string, or remove only the string
            subs_with_chunks = make_chunks(subs_with_silence, exclude = [],remove = ["#","\(.*?\)", "<.*?\]", "\[.*?\]", "<.*?>",  "\+/", "\+..."])
            
            with open(f"{output_path}/swedia/output/{name}.json", "w") as fout:
                simplejson.dump(subs_with_chunks, fout, indent=4, ignore_nan=True)

    


main()

