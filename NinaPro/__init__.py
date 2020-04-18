import os
from six.moves import urllib
import scipy.io as sio
import zipfile

#
# Responsible for downloading, and formatting data into "loaded_data"
#
class NinaPro:

    raw_data_dir    = "raw"
    proc_data_dir   = "processed"
    nina_data_urls  = {
                        "s1.zip": "https://zenodo.org/record/1000116/files/s1.zip?download=1",
                        "s2.zip": "https://zenodo.org/record/1000116/files/s2.zip?download=1",
                        "s3.zip": "https://zenodo.org/record/1000116/files/s3.zip?download=1",
                        "s4.zip": "https://zenodo.org/record/1000116/files/s4.zip?download=1",
                        "s5.zip": "https://zenodo.org/record/1000116/files/s5.zip?download=1",
                        "s6.zip": "https://zenodo.org/record/1000116/files/s6.zip?download=1",
                        "s7.zip": "https://zenodo.org/record/1000116/files/s7.zip?download=1",
                        "s8.zip": "https://zenodo.org/record/1000116/files/s8.zip?download=1",
                        "s9.zip": "https://zenodo.org/record/1000116/files/s9.zip?download=1",
                        "s10.zip": "https://zenodo.org/record/1000116/files/s10.zip?download=1"
                    }

    loaded_data = {}    # A dictionary with the following structure:
                        #
                        #       {
                        #           "s1": {
                        #                       "E1": {... },
                        #                       "E2": {... },
                        #                       "E3": {... }
                        #                   }
                        #
                        #           "s2:" {
                        #                       "E1": {... }, ...
                        #                   }
                        #
                        #           ...
                        #       }


    def __init__(self, all_data_path, grab_all_data = True):

        if all_data_path is None:
            raise ValueError("All data path is empty.")

        #
        # Create directory structure
        #
        if not os.path.exists(all_data_path):
            os.makedirs(all_data_path)

        self.raw_data_path   = os.path.join(all_data_path, self.raw_data_dir)
        self.proc_data_path  = os.path.join(all_data_path, self.proc_data_dir)

        if not os.path.exists(self.raw_data_path):
            os.makedirs(self.raw_data_path)
        if not os.path.exists(self.proc_data_path):
            os.makedirs(self.proc_data_path)

        #
        # Obtain Nina dataset
        #
        if grab_all_data:
            if self.miss_proc_data():
                if self.miss_raw_data():
                    self.get_raw_data()
                self.process_raw_data()


    def load_processed_data(self):

        if self.miss_proc_data():
            raise RuntimeError("Missing processed data, unable to load processed data.")

        for subdir, dirs, files in os.walk(self.proc_data_path):
            for file in files:
                if file.endswith('.mat'):
                    file_path = os.path.join(subdir, file)

                    patient = subdir.split('/')[-1]
                    if patient not in self.loaded_data:
                        self.loaded_data[patient] = {}

                    _, exercise, _ = file.split('.')[0].split('_')
                    self.loaded_data[patient][exercise] = sio.loadmat(file_path)

        return self.loaded_data


    def get_raw_data(self):
        """
            Downloads each missing zip file into the "raw" directory.
        """

        zip_files   = list(self.nina_data_urls.keys())

        for i, zip in enumerate(zip_files):
            cur_path = os.path.join(self.raw_data_path, zip)

            if not os.path.exists(cur_path):
                cur_url         = self.nina_data_urls[zip]
                http_request    = urllib.request.urlopen(cur_url)

                if http_request is None:
                    raise RuntimeError("Unable to open the following url \n{}".format(cur_url))
                else:
                    print("{}/{}. Downloading \"{}\".".format(i + 1, len(zip_files), cur_url))

                with open(cur_path, "wb") as f:
                    f.write(http_request.read())


    def process_raw_data(self):
        """
            Extracts each zip file into the "processed" directory.
        """

        for zip in self.nina_data_urls.keys():

            cur_proc_path   = os.path.join(self.proc_data_path, zip.replace(".zip", ""))

            if not os.path.exists(cur_proc_path):
                cur_zip_path    = os.path.join(self.raw_data_path, zip)
                zip_ref         = zipfile.ZipFile(cur_zip_path, 'r')

                zip_ref.extractall(self.proc_data_path)
                zip_ref.close()

    def miss_proc_data(self):
        proc_data_miss = False

        for zip in self.nina_data_urls.keys():
            cur_proc_path = os.path.join(self.proc_data_path, zip.replace(".zip", ""))
            if not os.path.exists(cur_proc_path):
                proc_data_miss = True

        return proc_data_miss

    def miss_raw_data(self):
        raw_data_miss = False

        for zip in self.nina_data_urls.keys():
            cur_path = os.path.join(self.raw_data_path, zip)
            if not os.path.exists(cur_path):
                raw_data_miss = True

        return raw_data_miss