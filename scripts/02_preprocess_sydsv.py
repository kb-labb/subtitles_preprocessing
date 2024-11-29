import argparse
import pandas as pd
import simplejson
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

    colnames = ['sub_id', 'sound_file_id', 'sound_file_id2', 'speaker_id', 'sub_start', 'sub_end', 'text']
    df = pd.read_csv(csv_file, sep = '|', names=colnames)
    df.set_index('sub_id', inplace=True)

    g = df.groupby('sound_file_id')
    audio_files = list(g.groups.keys())
    
    total_duration = 0
    for name in audio_files:
        subs = g.get_group(name)
        audio_path = path + "/sydsvenska-knippen-mp3/" + name + ".mp3"
        subs_dict = pandas_to_dict(subs, audio_path)

        subs_with_silence = add_silence(subs_dict)

        #exclude entire sub containing substring, or remove only the substring
        subs_with_chunks = make_chunks(subs_with_silence, exclude = ['/', '(', '«','»'], remove = ["\.\.\."])

        #with open(f"{path}/output/{name}.json", "w") as fout:
        with open(path + '/output/'+name+".json", "w") as fout:
            simplejson.dump(subs_with_chunks, fout, indent=4, ignore_nan=True)

        

main()
