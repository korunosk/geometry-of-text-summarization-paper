import os
import shutil
import re
import random
import json
import wget
import numpy as np


TMP_DIR  = 'tmp'
DATA_DIR = 'data'

URLS = [
	'https://drive.google.com/u/0/uc?id=1z1_i3cCQOd-1PWfaoFwO34YgCvdJemH7&export=download',
	'https://github.com/neulab/REALSumm/raw/master/scores_dicts/abs.pkl',
	'https://github.com/neulab/REALSumm/raw/master/scores_dicts/ext.pkl'
]

DOC_IDS = [
	 1017, 10586, 11343,  1521, 2736,
	 3789,  5025,  5272,  5576, 6564,
	 7174,  7770,  8334,  9325, 9781,
	10231, 10595, 11351,  1573, 2748,
	 3906,  5075,  5334,  5626, 6714,
	 7397,  7823,  8565,  9393, 9825,
	10325, 10680, 11355,  1890, 307,
	 4043,  5099,  5357,  5635, 6731,
	 7535,  7910,  8613,  9502, 10368,
	10721,  1153,    19,  3152, 4303,
	 5231,  5420,  5912,  6774, 7547,
	 8001,  8815,  9555, 10537, 10824,
	 1173,  1944,  3172,  4315, 5243,
	 5476,  6048,  6784,  7584, 8054,
	 8997,  9590, 10542, 11049, 1273,
	 2065,  3583,  4637,  5244, 5524,
	 6094,  6976,  7626,  8306, 9086,
	 9605, 10563, 11264,  1492, 2292,
	 3621,  4725,  5257,  5558, 6329,
	 7058,  7670,  8312,  9221, 9709
]


def preprocess_document(d):
	d = d.strip()
	d = d.split(' . ')
	for i in range(len(d) - 1):
		d[i] += ' .'
	return d


def preprocess_summary(s):
	s = re.findall(r'<t>(.*?)</t>', s)
	s = list(map(lambda x: x.strip(), s))
	return s


def make_doc_id(i):
	return f'D{str(i).zfill(5)}'


def make_annotations(data, dataset):
	for d in dataset.values():
		sdoc_id = make_doc_id(DOC_IDS[d['doc_id']])
		
		ref_summ = preprocess_summary(d['ref_summ'])
		
		data[sdoc_id]['references'].append([None, ref_summ])
		
		for ss in d['system_summaries'].values():
			system_summary = preprocess_summary(ss['system_summary'])
			
			data[sdoc_id]['annotations'].append({
				"topic_id": None,
				"summ_id": None,
				"pyr_score": ss['scores']['litepyramid_recall'],
				"pyr_mod_score": None,
				"text": system_summary,
				"responsiveness": None
			})
	
	return data


def fix_annotations(data):
	for doc_id in DOC_IDS:
		sdoc_id = make_doc_id(doc_id)
		
		# Shuffle the annotations
		random.shuffle(data[sdoc_id]['annotations'])
		
		# Set the summary ID parameter
		for i, a in enumerate(data[sdoc_id]['annotations']):
			a['summ_id'] = str(i)
			
	return data


if not os.path.exists(TMP_DIR):
	print(f'"{TMP_DIR}" does not exist. Creating it.')
	os.makedirs(TMP_DIR)

for url in URLS:
	filename = wget.download(url, out=TMP_DIR)
	print(f'Downloaded {filename}')

with open(f'{TMP_DIR}/src.txt', mode='r') as fp:
	src = fp.readlines()

ext_pkl = np.load(f'{TMP_DIR}/ext.pkl', allow_pickle=True)
abs_pkl = np.load(f'{TMP_DIR}/abs.pkl', allow_pickle=True)

data = {}

for doc_id in DOC_IDS:
	sdoc_id = make_doc_id(doc_id)
	
	data[sdoc_id] = {
		'documents': [],
		'references': [],
		'annotations': []
	}

	document = preprocess_document(src[doc_id])
	
	data[sdoc_id]['documents'].append(document)

data = make_annotations(data, ext_pkl)
data = make_annotations(data, abs_pkl)
data = fix_annotations(data)

with open(f'{DATA_DIR}/CNNDM.json', mode='w') as fp:
	json.dump(data, fp, indent=4)
	print('CNNDM file saved')

if os.path.exists(TMP_DIR):
	print(f'Removing {TMP_DIR}')
	shutil.rmtree(TMP_DIR)
