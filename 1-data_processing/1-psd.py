import mne
import numpy as np
import os


def eeg_power_band(epochs):
    """
    根据epochs的特定频段中的相对功率来创建EEG特征。
    """
    FREQ_BANDS = {"delta": [0.5, 4],
                  "theta": [4, 8],
                  "alpha": [8, 13],
                  "sigma": [13, 25],
                  "beta": [25, 45]}
    
    spectrum = epochs.compute_psd(method='welch', fmin=0.5, fmax=45., n_fft=256, n_overlap=10)
    psds, freqs = spectrum.get_data(return_freqs=True)
    
    psds /= np.sum(psds, axis=-1, keepdims=True)
    
    X = []
    for fmin, fmax in FREQ_BANDS.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        # psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)]
        X.append(psds_band.reshape(len(psds), -1))
    
    return np.concatenate(X, axis=1)




def process_eeg_data(input_folders, output_folders):
    """
    处理输入文件夹中的所有.set文件，并将提取的特征保存到输出文件夹中。
    """
    for input_folder, output_folder in zip(input_folders, output_folders):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(".set"):
                set_file_path = os.path.join(input_folder, filename)
                
                # Load the .set file using MNE
                raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
                
                raw.resample(512)
                # Set EEG reference
                raw.set_eeg_reference('average', projection=True)
                # Filter the data
                raw.filter(1., 45., fir_design='firwin')
                
                # Create events (you may need to modify this based on your data)
                events = mne.make_fixed_length_events(raw, start=0, duration=5.0)
                
                # Create Epochs object
                epochs = mne.Epochs(raw, events, tmin=0, tmax=4.0, baseline=None, preload=True)
                # print('epochs-info',epochs.info)
                # # Get original events data
                # original_events = epochs.get_data()
                # Extract features
                # print('original_events',original_events.shape)
                
                features = eeg_power_band(epochs)
                
                # Save features to output folder
                output_file_path = os.path.join(output_folder, filename.replace(".set", "_features.npy"))
                print(features.shape)
                np.save(output_file_path, features)
                
                print(f"Processed {filename} and saved features to {output_file_path}")
def AD_CN():
    # Example usage
    input_folders = ["psd_AD_CN/test/AD", "psd_AD_CN/test/CN",'psd_AD_CN/train/AD','psd_AD_CN/train/CN']
    output_folders = ["psd/test/AD", "psd//test/CN",'psd/train/AD','psd/train/CN']
    process_eeg_data(input_folders, output_folders)

def CN_FDT():
    # Example usage
    input_folders = ["psd_CN_FTD/test/FTD", "psd_CN_FTD/test/CN",'psd_CN_FTD/train/FTD','psd_CN_FTD/train/CN']
    output_folders = ["psd/test/FTD", "psd//test/CN",'psd/train/FTD','psd/train/CN']
    process_eeg_data(input_folders, output_folders)

def main():

    # AD_CN()
    CN_FDT()

if __name__ == '__main__':
    main()