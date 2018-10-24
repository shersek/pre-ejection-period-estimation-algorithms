import os
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt

#sampling rate of training dataset
F_SAMPLING=2000

class TrainingSubject(object):
    """
        Instance class represents a training subject
    """
    def __init__(self, dir, subject_id , store_in_ram=False):
        self.dir = dir
        self.subject_id = subject_id
        self.subject_file_name = self._get_subject_file_name()
        self.rest_interval = self._load_rest_intervals()
        self.recovery_interval=self._load_recovery_intervals()
        self.store_in_ram=store_in_ram

        if store_in_ram:
            self.aX = self.get_aX()
            self.aY = self.get_aY()
            self.aZ = self.get_aZ()
            self.gX = self.get_gX()
            self.gY = self.get_gY()
            self.gZ = self.get_gZ()
            self.ecg = self.get_ecg()
            self.bcg = self.get_bcg()
            self.icg = self.get_icg()
            self.bp = self.get_bp()
        else:
            self.aX = None
            self.aY = None
            self.aZ = None
            self.gX = None
            self.gY = None
            self.gZ = None
            self.ecg = None
            self.bcg = None
            self.icg = None
            self.bp = None

    def _get_subject_file_name(self ):
        '''
        :return: filename for the .mat file containing training subject data
        '''
        return 'Filtered_Subject_' + str(self.subject_id) + '_Mid_Sternum_Rest_Exer_Rec.mat'


    def _load_rest_intervals(self):
        '''
        :return: list contatining start and end of resting interval for the training subject, pulled from subject_info_training.csv
        '''
        df = pd.read_csv(self.dir + '/subject_info_training.csv')
        return [F_SAMPLING*int(df['REST START'][df['SUBJECT ID'] == self.subject_id]) , F_SAMPLING*int(df['REST END'][df['SUBJECT ID'] == self.subject_id])]

    def _load_recovery_intervals(self):
        '''
        :return: list contatining start and end of recovery interval for the training subject, pulled from subject_info_training.csv
        '''
        df = pd.read_csv(self.dir + '/subject_info_training.csv')
        return [F_SAMPLING*int(df['RECOVERY START'][df['SUBJECT ID'] == self.subject_id]) , F_SAMPLING*int(df['RECOVERY END'][df['SUBJECT ID'] == self.subject_id])-1  ]

    def get_ecg(self):
        '''
        :return: ecg signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['ecg_filtered'].reshape(-1)


    def get_aX(self ):
        '''
        :return: aX signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['ax_f'].reshape(-1)

    def get_aY(self):
        '''
        :return: aY signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['ay_f'].reshape(-1)

    def get_aZ(self):
        '''
        :return: aZ signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['az_f'].reshape(-1)

    def get_icg(self):
        '''
        :return: impedance cardiography signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['icg_filtered'].reshape(-1)

    def get_bp(self):
        '''
        :return: blood pressure signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)


        if 'bp_filtered' in mat_contents: #blood pressure exists only for some subjects
            return mat_contents['bp_filtered'].reshape(-1)
        else:
            return None

    def get_bcg(self):
        '''
        :return: bcg signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['bcg_filtered'].reshape(-1)


    def get_gX(self ):
        '''
        :return: gX signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['gyro_x_filtered'].reshape(-1)

    def get_gY(self):
        '''
        :return: gY signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['gyro_y_filtered'].reshape(-1)

    def get_gZ(self):
        '''
        :return: gZ signal for the subject as a numpy vector
        '''
        mat_contents = sio.loadmat(self.dir + '/' + self.subject_file_name)

        return mat_contents['gyro_z_filtered'].reshape(-1)


def load_subjects(dir , store_in_ram=False):
    '''
    load all training subject instances into a list
    :param dir: directory of training files
    :param store_in_ram: store all signals in RAM ? much faster training but consumes lots of memory
    :return: list of training subject instances
    '''
    unique_subjects = [int(file_name[17:20]) for file_name in
                       os.listdir(dir) if
                       file_name.startswith('Filtered')]
    subjects = []
    for subject_id in unique_subjects:
        subjects.append(TrainingSubject(dir, subject_id , store_in_ram))
    return subjects

def diagnose_training_subjects(list_of_subjects):
    '''
    plot bcg signals from list of subjects for debugging
    :param list_of_subjects: list of subject instances
    :return:
    '''
    for subject in list_of_subjects:
        plt.figure()
        plt.plot(subject.bcg)
        plt.plot(subject.rest_interval, subject.bcg[subject.rest_interval], 'ok')
        plt.plot(subject.recovery_interval, subject.bcg[subject.recovery_interval], 'ok')
        plt.title(str(subject.subject_id))



class TestingRecordingtHF(object):

    def __init__(self, dir, subject_id, recording_id, store_in_ram=False):
        self.dir = dir
        self.subject_id = subject_id
        self.recording_id = recording_id
        self.recording_file_name = self._get_recording_file_name()
        self.store_in_ram = store_in_ram

        if store_in_ram:
            self.aX = self.get_aX()
            self.aY = self.get_aY()
            self.aZ = self.get_aZ()
            self.ecg = self.get_ecg()
            self.bcg = self.get_bcg()
        else:
            self.aX = None
            self.aY = None
            self.aZ = None
            self.ecg = None
            self.bcg = None

    def _get_recording_file_name(self):

        return 'trm' + str(self.subject_id) + '.01.rec' + str(self.recording_id) + '.scale_wearable_filtered.mat'


    def get_ecg(self):

        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name)

        return -mat_contents['ecg_wearable_filtered'].reshape(-1)

    def get_aX(self):
        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name)

        return -mat_contents['acc_y_filtered'].reshape(-1) #swap x and y #invert x

    def get_aY(self):
        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name) #swap x and y

        return mat_contents['acc_x_filtered'].reshape(-1)

    def get_aZ(self):
        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name)

        return mat_contents['acc_z_filtered'].reshape(-1)

    def get_bcg(self):
        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name)

        return mat_contents['bcg_filtered'].reshape(-1)

    def get_ecg_scale(self):
        mat_contents = sio.loadmat(self.dir + '/' + self.recording_file_name)

        return mat_contents['ecg_scale_filtered'].reshape(-1)


def load_testing_hf_subjects(dir, store_in_ram=False):

    list_recordings =  os.listdir(dir)
    recordings = []
    for recording_name in list_recordings:
        recordings.append(TestingRecordingtHF(dir, int(recording_name[3:6]) , int(recording_name[13]) , store_in_ram))
    return recordings


