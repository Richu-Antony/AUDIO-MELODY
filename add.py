#@title #Part2: Setup
#@markdown This cell needs to be run only once. It will mount your Google Drive and setup prerequisites.<br><br>
#@markdown <small>Mounting Drive will enable this notebook to save outputs directly to your Drive. Otherwise you will need to copy/download them manually from this notebook.</small><br><br>

force_setup = False
repositories = ['https://github.com/haoheliu/AudioLDM.git']
pip_packages = ''
apt_packages = ''
mount_drive = True #@param {type:"boolean"}
skip_setup = False #@ param {type:"boolean"}
local_models_dir = "Music" #@param {type:"string"}

use_checkpoint = "audioldm-full-s-v2" #@param ["audioldm-s-full", "audioldm-full-l", "audioldm-full-s-v2"]


if use_checkpoint == 'audioldm-s-full':
  ckpt_url = 'https://zenodo.org/record/7600541/files/'+use_checkpoint+'.ckpt?download=1'
else:
  ckpt_url = 'https://zenodo.org/record/7698295/files/'+use_checkpoint+'.ckpt?download=1'
  
use_ckpt = use_checkpoint+'.ckpt'


import os
from google.colab import output
import warnings
warnings.filterwarnings('ignore')
%cd /content/

if pip_packages != '':
  !pip -q install {pip_packages}
if apt_packages != '':
  !apt-get update && apt-get install {apt_packages}

import sys, time, ntpath, string, random, librosa, librosa.display, IPython, shutil, math, psutil, datetime, requests, pytz
import numpy as np
import soundfile as sf
from datetime import timedelta

# Print colors
class c:
  title = '\033[96m'
  ok = '\033[92m'
  okb = '\033[94m'
  warn = '\033[93m'
  fail = '\033[31m'
  endc = '\033[0m'
  bold = '\033[1m'
  dark = '\33[90m'
  u = '\033[4m'

def op(typex, msg, value='', time=False):
  if time == True:
    stamp = timestamp(human_readable=True)
    typex = c.dark+stamp+' '+typex
  if value != '':
    print(typex+msg+c.endc, end=' ')
    print(value)
  else:
    print(typex+msg+c.endc)

def gen_id(type='short'):
  id = ''
  if type == 'timestamp':
    id = timestamp()
  if type == 'short':
    id = requests.get('https://api.inha.asia/k/?type=short').text
  if type == 'long':
    id = requests.get('https://api.inha.asia/k').text
  return id

def timestamp(no_slash=False, human_readable=False, helsinki_time=True, date_only=False):
  if helsinki_time == True:
    dt = datetime.datetime.now(pytz.timezone('Europe/Helsinki'))
  else:
    dt = datetime.datetime.now()
  if no_slash == True:
    dt = dt.strftime("%Y%m%d%H%M%S")
  else:
    if human_readable == True:
      dt = dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
      if date_only == True:
        dt = dt.strftime("%Y-%m-%d")
      else:
        dt = dt.strftime("%Y-%m-%d_%H%M%S")
  return dt;

def fix_path(path, add_slash=False):
  if path.endswith('/'):
    path = path #path[:-1]
  if not path.endswith('/'):
    path = path+"/"
  if path.startswith('/') and add_slash == True:
    path = path[1:]
  return path
  
def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail or ntpath.basename(head)

def path_dir(path):
  return path.replace(path_leaf(path), '')

def path_ext(path, only_ext=False):
  filename, extension = os.path.splitext(path)
  if only_ext == True:
    extension = extension[1:]
  return extension

def basename(path):
  filename = os.path.basename(path).strip()#.replace(" ", "_")
  filebase = os.path.splitext(filename)[0]
  return filebase

def slug(s):
  valid_chars = "-_. %s%s" % (string.ascii_letters, string.digits)
  file = ''.join(c for c in s if c in valid_chars)
  file = file.replace(' ','_')
  return file
  
def fetch(url, save_as):
  headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
  try:
    r = requests.get(url, stream=True, headers=headers, timeout=5)
    if r.status_code == 200:
      with open(save_as, 'wb') as f:
        r.raw.decode_content = True
        shutil.copyfileobj(r.raw, f)
      resp = r.status_code
    else:
      resp = 0
  except requests.exceptions.ConnectionError as e:
    r = 0
    resp = r
  return resp

def list_audio(path, midi=False):
  audiofiles = []
  for ext in ('*.wav', '*.aiff', '*.aif', '*.caf' '*.flac', '*.mp3', '*.m4a', '*.ogg', '*.WAV', '*.AIFF', '*.AIF', '*.CAF', '*.FLAC', '*.MP3', '*.OGG'):
    audiofiles.extend(glob(join(path, ext)))
  if midi is True:
    for ext in ('*.mid', '*.midi', '*.MID', '*.MIDI'):
      audiofiles.extend(glob(join(path, ext)))
  audiofiles.sort()
  return audiofiles

def audio_player(input, sr=44100, limit_duration=2):
  if type(input) != np.ndarray:
    input, sr = librosa.load(input, sr=None, mono=False)
  if limit_duration > 0:
    last_sample = math.floor(limit_duration*60*sr)
    if input.shape[-1] > last_sample:
      input = input[:last_sample, :last_sample]
      op(c.warn, 'WARN! Playback of below audio player is limited to first '+str(limit_duration)+' minutes to prevent Colab from crashing.\n')
  IPython.display.display(IPython.display.Audio(input, rate=sr))

# Mount Drive
if mount_drive == True:
  if not os.path.isdir('/content/drive'):
    from google.colab import drive
    drive.mount('/content/drive')
    drive_root = '/content/drive/My Drive/'
  if not os.path.isdir('/content/mydrive'):
    os.symlink('/content/drive/My Drive', '/content/mydrive')
    drive_root = '/content/mydrive/'
  drive_root_set = True
else:
  os.mkdir('/content/faux_drive')
  drive_root = '/content/faux_drive/'

if mount_drive == False:
  local_models_dir = ''

if len(repositories) > 0 and skip_setup == False:
  for repo in repositories:
    %cd /content/
    install_dir = fix_path('/content/'+path_leaf(repo).replace('.git', ''))
    repo = repo if '.git' in repo else repo+'.git'
    !git clone {repo}
    if os.path.isfile(install_dir+'setup.py') or os.path.isfile(install_dir+'setup.cfg'):
      !pip install -e {install_dir}
    if os.path.isfile(install_dir+'requirements.txt'):
      !pip install -r {install_dir}/requirements.txt

if len(repositories) == 1:
  %cd {install_dir}

dir_tmp = '/content/tmp/'
if not os.path.isdir(dir_tmp): os.mkdir(dir_tmp)

use_ckpt_path = os.path.expanduser('~')+'/.cache/audioldm/'

if not os.path.isdir(use_ckpt_path):
  os.makedirs(use_ckpt_path)

if local_models_dir != '':
  models_dir = drive_root+fix_path(local_models_dir)
  if not os.path.isdir(models_dir):
    os.makedirs(models_dir)
  # for ckpt_url in ckpt_urls:
  #   use_ckpt = ckpt_url.split('files/')[1].split('?')[0]
  if os.path.isfile(models_dir+use_ckpt):
    op(c.title, 'Fetching local ckpt:', models_dir.replace(drive_root, '')+use_ckpt)
    shutil.copy(models_dir+use_ckpt, use_ckpt_path+use_ckpt)
    op(c.ok, 'Done.')
  else:
    op(c.warn, 'Downloading '+use_ckpt+' to ', models_dir.replace(drive_root, ''))
    !wget {ckpt_url} -O {models_dir}{use_ckpt}
    shutil.copy(models_dir+use_ckpt, use_ckpt_path+use_ckpt)
    op(c.ok, 'Done.')
else:
  # for ckpt_url in ckpt_urls:
  #   use_ckpt = ckpt_url.split('files/')[1].split('?')[0]
  models_dir = use_ckpt_path
  op(c.warn, 'Downloading', use_ckpt)
  !wget {ckpt_url} -O {models_dir}{use_ckpt}
  shutil.copy(models_dir+use_ckpt, use_ckpt_path+use_ckpt)
  op(c.ok, 'Done.')

ckpt_path = use_ckpt_path+use_ckpt
op(c.title, 'Build model', ckpt_path)
sys.path.append('/content/AudioLDM/audioldm/')
from audioldm import text_to_audio, style_transfer, super_resolution_and_inpainting, build_model, latent_diffusion
audioldm = build_model(ckpt_path=ckpt_path, model_name=use_checkpoint)

def round_to_multiple(number, multiple):
  x = multiple * round(number / multiple)
  if x == 0: x = multiple
  return x

def text2audio(text, duration, audio_path, guidance_scale, random_seed, n_candidates, steps):
  waveform = text_to_audio(
    audioldm,
    text,
    audio_path,
    random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    ddim_steps=steps,
    n_candidate_gen_per_text=int(n_candidates)
  )
  if(len(waveform) == 1):
    waveform = waveform[0]
  return waveform

def styleaudio(text, duration, audio_path, strength, guidance_scale, random_seed, steps):
  waveform = style_transfer(
    audioldm,
    text,
    audio_path,
    strength,
    random_seed,
    duration=duration,
    guidance_scale=guidance_scale,
    ddim_steps=steps,
  )
  if(len(waveform) == 1):
    waveform = waveform[0]
  return waveform


# time_mask_ratio_start_and_end=(0.10, 0.15), # regenerate the 10% to 15% of the time steps in the spectrogram
# time_mask_ratio_start_and_end=(1.0, 1.0), # no inpainting
# freq_mask_ratio_start_and_end=(0.75, 1.0), # regenerate the higher 75% to 100% mel bins
# freq_mask_ratio_start_and_end=(1.0, 1.0), # no super-resolution
def superres(text, duration, audio_path, guidance_scale, random_seed, n_candidates, steps):
  waveform = super_resolution_and_inpainting(
    audioldm,
    text,
    audio_path,
    random_seed,
    ddim_steps=steps,
    duration=duration,
    guidance_scale=guidance_scale,
    n_candidate_gen_per_text=n_candidates,
    freq_mask_ratio_start_and_end=(0.75, 1.0)
  )
  if(len(waveform) == 1):
    waveform = waveform[0]
  return waveform


prompt_list = []

output.clear()
# !nvidia-smi
print()
op(c.title, 'Using:', use_ckpt, time=True)
op(c.ok, 'Setup finished.', time=True)
print()