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


required = {"librosa", "numpy", "hmmlearn", "sklearn", "pathlib", "matplotlib", "seaborn"}
installIfNotExist(required)

# silence warning
warnings.simplefilter('ignore')
logging.getLogger("hmmlearn").setLevel("CRITICAL") # silence warning for models

################################Data analysis and visulization####################################

def visualize_1():
    import librosa.display
    import matplotlib.pyplot as plt

    # Define parameters
    n_fft = 256
    hop_length = 256
    n_mels = 12

    # Load audio file
    audio_file = 'train/1/1.wav'
    y, sr = librosa.load(audio_file)

    # Compute Mel filterbank
    mel_filterbank = librosa.filters.mel(sr=8000, n_fft=n_fft, n_mels=n_mels)

    # Visualize Mel filterbank as a heatmap
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_filterbank, x_axis='linear', cmap='coolwarm')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Mel bin')
    plt.title('Mel filterbank')
    plt.tight_layout()
    plt.show()



def visualize_2():
    import librosa.display
    import matplotlib.pyplot as plt

    # Load audio file
    audio_file = 'train/1/1.wav'
    y, sr = librosa.load(audio_file)

    # Compute MFCC coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=8000, n_mfcc=12)

    # Visualize MFCC coefficients as a heatmap
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', cmap='coolwarm')
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()


def show_data_sets():
    import numpy as np
    import matplotlib.pyplot as plt

    # Define the training sets and files for each GMM-HMM model
    training_sets = {f"train/{i}" for i in range(1, 17)}
    training_files = [['5.wav', '4.wav', '1.wav', '3.wav', '2.wav']]*16
    training_files[13] = ['6.wav', '7.wav', '5.wav', '4.wav', '1.wav', '3.wav', '2.wav']

    N_states = [6]*80
    for i in (12, 13, 14, 15, 16):
        start_index = (i - 1) * 5
        end_index = start_index + 7
        N_states[start_index:end_index] = [3]*7

    train_stats = {}
    for i, files in enumerate(training_files, start=1):
        stats = {}
        for file in files:
            stats[file] = N_states[(i-1)*5 + files.index(file)]
        train_stats[f"train/{i}"] = stats

    fig, axs = plt.subplots(4, 4, figsize=(12, 10), sharex=True, sharey=True)
    fig.text(0.04, 0.5, 'N State', va='center', rotation='vertical')
    for i, (training_set, stats) in enumerate(train_stats.items()):
        row = i // 4
        col = i % 4
        ax = axs[row, col]
        ax.bar(list(stats.keys()), list(stats.values()), width=0.5, align='center')
        ax.set_title(training_set)
        ax.tick_params(axis='x', rotation=90)
    plt.suptitle('Distribution of Number of States for GMM-HMM Models', fontsize=16, y=1.05)
    plt.show()

def plot_test_result(ax, test_files, det_lab, decode):
    matched_labels_dict = {test_files[i] : det_lab[i] for i in range(len(test_files)) if int(test_files[i] // 7) + 1 == det_lab[i]}
    failed_labels_dict = {test_files[i] : det_lab[i] for i in range(len(test_files)) if int(test_files[i] // 7) + 1 != det_lab[i]}
    
    ax.scatter(matched_labels_dict.keys(), matched_labels_dict.values(), label="Matched Test Audio", color="blue")
    ax.scatter(failed_labels_dict.keys(), failed_labels_dict.values(), label="Failed Test Audio", color="red")
    ax.legend()

    ax.set_xlabel('Test Audio #')
    ax.set_ylabel('Detected Label')

    # Add the decode value as a text box
    ax.text(0.8, 0.2, f"Acc = {round(decode, 3)}%", ha='center', va='center', transform=ax.transAxes, fontsize=12)


def plot_confusion_matrix(true_lab, det_lab):
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(true_lab, det_lab)

    # Plot the confusion matrix
    classes = [f"train/{i}" for i in range(1, 17)]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot()
    plt.xticks(rotation=90)
    plt.show()


#######################################################################################################

import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
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

    test_files = []
    true_lab = []
    det_lab = []

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
        
        test_files.append(i)
        true_lab.append(lab_true)
        det_lab.append(int(lab_det))
        print("true lab  %d det lab %d" % (lab_true, lab_det))
    
    decode = count * 100 / (test_end - test_start)
    print("decode  %.2f   " % decode)
    
    return test_files, true_lab, det_lab, decode


if __name__ == "__main__":
    # visulization the training dataset
    # show_data_sets()

    train_dir = "train"
    test_dir = "test"

    # Generate model from the training data
    all_models = generate_model_from_data(train_dir)

    # save models to file for development 
    # np.save("models_hmm.npy", all_models)
    # all_models = np.load("models_hmm.npy", allow_pickle=True)

    # Test the model
    # 1-11 is two words
    test_files2, true_lab2, det_lab2, decode2 = test_model(test_dir, all_models[0: 11], 0, 10)
    # 12-16 is single word
    test_files1, true_lab1, det_lab1, decode1 = test_model(test_dir, all_models[11: 16], 11, 16)

    # plot the test results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    ax1.set_title('Double-Words')
    ax2.set_title('Single-Words')
    plot_test_result(ax1, test_files2, det_lab2, decode2)
    plot_test_result(ax2, test_files1, det_lab1, decode1)
    plt.show()

    # show the confusion matrix

    # Get the true labels
    true_lab = true_lab2 + true_lab1
    det_lab = det_lab2 + det_lab1
    plot_confusion_matrix(true_lab, det_lab)
