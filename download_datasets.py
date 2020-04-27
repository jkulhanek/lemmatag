#  
# MIT License
# 
# Copyright (c) 2018 Dan Kondratyuk
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# 
# Downloads universal dependencies datasets
# See http://universaldependencies.org for a list of all datasets


from tqdm import tqdm
import requests
import math
import tarfile
import os


def download_file(url, filename=None):
    # Streaming, so we can iterate over the response
    r = requests.get(url, stream=True)

    # Total size in bytes
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024
    wrote = 0

    with open(filename, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size // block_size),
                         unit='KB',
                         unit_scale=True):
            wrote = wrote + len(data)
            f.write(data)

    if total_size != 0 and wrote != total_size:
        print("Error, something went wrong")


def extract_file(read_filename, output_path):
    tar = tarfile.open(read_filename, 'r')
    tar.extractall(output_path)



# Converts CoNLL format of Universal Dependency (UD) files to LemmaTag format
# See http://universaldependencies.org/format.html

column_names = [
    "ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC"
]

column_pos = {name: i for i, name in enumerate(column_names)}


def conllu_to_lemmatag(lines, pos_column="XPOS", max_lines=None):
    line_count = 0

    for line in lines:
        line = line.strip()
        if line.startswith("#"):
            continue
        elif line == "":
            line_count = 0
            yield ""
        else:
            if max_lines and line_count and line_count >= max_lines:
                continue

            line_count += 1
            tokens = line.split("\t")
            yield "\t".join([tokens[column_pos["FORM"]], tokens[column_pos["LEMMA"]], tokens[column_pos[pos_column]]])

def convert_dataset(path):
    allfiles = []
    for root, dirs, files in os.walk(path, topdown=False):
        for filename in files:
            fname, ext = os.path.splitext(filename)
            if ext == '.conllu':
                allfiles.append(os.path.join(root, filename))

    for filename in tqdm(allfiles): 
        fname, ext = os.path.splitext(filename)
        writepath = fname + '.lemmatag'
        with open(filename, 'r') as fr, open(writepath, 'w+') as fw:
            fw.writelines(x + '\n' for x in conllu_to_lemmatag(fr))

def download_dataset(data_folder):
    dataset_url = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-2837/ud-treebanks-v2.2.tgz" \
                  "?sequence=1&isAllowed=y"
    dataset_path = os.path.join(data_folder, "ud-treebanks-v2.2.tgz")

    print("Downloading dataset")
    #download_file(dataset_url, dataset_path)

    print("Extracting dataset")
    #extract_file(dataset_path, data_folder)

    print("Converting to LemmaTag format")
    dataset_path = os.path.join(data_folder, 'ud-treebanks-v2.2') 
    convert_dataset(dataset_path)

    print("Downloaded successfully")

def ensure_dataset_exists(data_folder):
    dataset_path = os.path.join(data_folder, 'ud-treebanks-v2.2') 
    if not os.path.exists(dataset_path):
        download_dataset(data_folder) 
        return True

if __name__ == "__main__":
    data_folder = os.environ['DATASETS_PATH'] if 'DATASETS_PATH' in os.environ else os.path.expanduser('~/datasets')
    #if not ensure_dataset_exists(data_folder):
    #    print('Dataset already exists')
    download_dataset(data_folder)


