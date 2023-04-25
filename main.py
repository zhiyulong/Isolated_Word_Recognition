############################ Check and setup running enviroment at first ###############################
import logging
import pkg_resources
import subprocess
import sys
import warnings

# check if the packages are installed
def installIfNotExist(required):
  installed = {pkg.key for pkg in pkg_resources.working_set}
  missing = required - installed

  if missing:
    print(">> Installing the missing packages: ", missing)
    subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + list(missing), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    print("\n--Finished checking required packages.--\n")
  else:
    print("\n--Required packages already satisfied.--\n")


required = {"librosa", "numpy", "hmmlearn", "sklearn", "pathlib"}
installIfNotExist(required)

warnings.simplefilter('ignore') # silence warning
logging.getLogger("hmmlearn").setLevel("CRITICAL") # silence warning for models
#######################################################################################################

import librosa
import numpy as np
import os
from pathlib import Path
from hmmlearn.hmm import GMMHMM
from sklearn.cluster import KMeans

def init_para_hmm(collect_fea, N_state, N_mix):
    pi = np.zeros(N_state)
    pi[0] = 1
    A = np.zeros([N_state, N_state])
    for i in range(N_state - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5
    A[-1, -1] = 1

    feas = collect_fea
    len_feas = []
    for fea in feas:
        len_feas.append(np.shape(fea)[0])

    _, D = np.shape(feas[0])
    hmm_means = np.zeros([N_state, N_mix, D])
    hmm_sigmas = np.zeros([N_state, N_mix, D])
    hmm_ws = np.zeros([N_state, N_mix])
    for s in range(N_state):
        sub_fea_collect = []
        for fea, T in zip(feas, len_feas):
            T_s = int(T / N_state) * s
            T_e = int(T / N_state) * (s + 1)
            sub_fea_collect.append(fea[T_s:T_e])
        ws, mus, sigmas = gen_para_GMM(sub_fea_collect, N_mix)
        hmm_means[s] = mus
        hmm_sigmas[s] = sigmas
        hmm_ws[s] = ws
    return pi, A, hmm_means, hmm_sigmas, hmm_ws


def run_kmeans(dataset, K, m=20):
    """
    Applu K-Means for clustering data
    """
    labs = KMeans(n_clusters=K, random_state=9).fit_predict(dataset)
    return labs


def gen_para_GMM(fea_collect, N_mix):
    # use kmeans to group features
    feas = np.concatenate(fea_collect, axis=0)
    N, D = np.shape(feas)
    # initialize center
    labs = run_kmeans(feas, N_mix, m=20)
    mus = np.zeros([N_mix, D])
    sigmas = np.zeros([N_mix, D])
    ws = np.zeros(N_mix)
    for m in range(N_mix):
        index = np.where(labs == m)[0]
        # print("----index---------",index)
        sub_feas = feas[index]
        mu = np.mean(sub_feas, axis=0)
        sigma = np.var(sub_feas, axis=0)
        sigma = sigma + 0.0001
        mus[m] = mu
        sigmas[m] = sigma

        # print("------N  D-------",N,np.shape(index)[0])
        ws[m] = np.shape(index)[0] / N
    ws = (ws + 0.01) / np.sum(ws + 0.01)
    return ws, mus, sigmas


def creat_GMM(mus, sigmas, ws):
    gmm = dict()
    gmm['mus'] = mus
    gmm['sigmas'] = sigmas
    gmm['ws'] = ws
    return gmm


def init_hmm(collect_fea, N_state, N_mix):
    model_GMM_hmm = dict()

    # initialize from state 0
    pi = np.zeros(N_state)
    pi[0] = 1
    model_GMM_hmm["pi"] = pi

    # define Transition probability
    A = np.zeros([N_state, N_state])
    for i in range(N_state - 1):
        A[i, i] = 0.5
        A[i, i + 1] = 0.5
    A[-1, -1] = 1
    model_GMM_hmm["A"] = A

    feas = collect_fea
    len_feas = []
    for fea in feas:
        len_feas.append(np.shape(fea)[0])

    states = []
    for s in range(N_state):
        print("STATE ----------------", s)
        sub_fea_collect = []
        # distribute every state with features
        for fea, T in zip(feas, len_feas):
            T_s = int(T / N_state) * s
            T_e = (int(T / N_state)) * (s + 1)

            sub_fea_collect.append(fea[T_s:T_e])
        ws, mus, sigmas = gen_para_GMM(sub_fea_collect, N_mix)
        gmm = creat_GMM(mus, sigmas, ws)
        states.append(gmm)

    model_GMM_hmm["S"] = states

    return model_GMM_hmm


def extract_MFCC(wav_file):
    # set sr to 8000
    y, sr = librosa.load(wav_file, sr=8000)
    # extract fea
    fea = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12, n_mels=24, n_fft=256, win_length=256, hop_length=80, lifter=12)
    # Regularization
    mean = np.mean(fea, axis=1, keepdims=True)
    std = np.std(fea, axis=1, keepdims=True)
    fea = (fea - mean) / std
    # add first difference method
    fea_d = librosa.feature.delta(fea)
    fea = np.concatenate([fea.T, fea_d.T], axis=1)

    return fea


def generate_model_from_data(train_path):   
    models = None
    for path, dirs, files in os.walk(train_path):
        if (len(files) == 0):
            models = [None] * len(dirs)
            continue
        collect_fea = []
        len_feas = []
        for file in files:
            # find .wav and extract the feature
            if file.split(".")[-1] == "wav":
                wav_file = os.path.join(path, file)
                fea = extract_MFCC(wav_file)
                collect_fea.append(fea)
                len_feas.append(np.shape(fea)[0])
         
        # initialize model i
        i = int(path.split("/")[1]) - 1
        # how many states for a word and how many mix of a state
        # 1-11 is two words and 12-16 is single word
        N_state = 6
        N_mix = 3
        if (i + 1) in (12, 13, 14, 15, 16):
            print(f"Training GMM-HMM for {path} (single word, N_state = 3): {files}")
            N_state = 3
        else:
            print(f"Training GMM-HMM for {path} (double word, N_state = 6): {files}")

        pi, A, hmm_means, hmm_sigmas, hmm_ws = init_para_hmm(collect_fea, N_state, N_mix)
        train_GMMHMM = GMMHMM(n_components=N_state, n_mix=N_mix, covariance_type='diag', n_iter=90, tol=1e-5,
                              verbose=False, init_params="", params="tmcw", min_covar=0.0001)
        train_GMMHMM.startprob_ = pi
        train_GMMHMM.transmat_ = A
        train_GMMHMM.weights_ = hmm_ws
        train_GMMHMM.means_ = hmm_means
        train_GMMHMM.covars_ = hmm_sigmas
        train_GMMHMM.fit(np.concatenate(collect_fea, axis=0), np.array(len_feas))
        # store the model for current trainning set
        models[i] = train_GMMHMM
    return(models)

def test_model(test_dir, models, train_start, train_end):
    test_end = train_end * 7
    test_start = train_start * 7
    count = 0
    for i in range(test_start, test_end):
        wav_file = os.path.join(test_dir, str(i + 1) + ".wav")
        fea = extract_MFCC(wav_file)

        lab_true = int(i // 7) + 1
        scores = []
        for model in models:
            score, _ = model.decode(fea)
            scores.append(score)

        lab_det = np.argmax(scores) + 1 + train_start
        if lab_det == lab_true:
            count = count + 1

        print("true lab  %d det lab %d" % (lab_true, lab_det))
    print("decode  %.2f   " % (count * 100 / (test_end - test_start)))


if __name__ == "__main__":
    train_dir = "train"
    test_dir = "test"

    # Generate model from the training data
    all_models = generate_model_from_data(train_dir)

    # Test the model
    # 1-11 is two words
    test_model(test_dir, all_models[0: 11], 0, 10)
    # 12-16 is single word
    test_model(test_dir, all_models[11: 16], 11, 16)

    