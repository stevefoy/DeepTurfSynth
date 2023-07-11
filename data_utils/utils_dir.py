import io
import os
import shutil
import errno
import argparse

class readable_dir( argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))


class writable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        self.prospective_dir=values
        if not os.path.isdir(self.prospective_dir):
            self.check_outDir_exists()

        if os.access(self.prospective_dir, os.R_OK):
            setattr(namespace,  self.dest, self.prospective_dir)
        else:
            raise argparse.ArgumentTypeError(
                            "readable_dir:{0} is not a readable dir".format(prospective_dir))

    def check_outDir_exists(self):

        try:
            os.makedirs(self.prospective_dir)
            print("Created new directory")
        except OSError as e:
            if e.errno != errno.EEXIST:
                print('An exception on creating file')
                raise


class readable_file(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_file = values
        if not os.path.isfile(prospective_file):
            raise argparse.ArgumentTypeError("readable_file:{0} is not a valid path".format(prospective_file))
        if os.access(prospective_file, os.R_OK):
            setattr(namespace, self.dest, prospective_file)
        else:
            raise argparse.ArgumentTypeError("readable_file:{0} is not a readable dir".format(prospective_file)) 



class directory_editor(object):
    def __init__(self, inputDir, outputDir):
        self.inputDir= inputDir
        self.outputDir = outputDir

    def check_outDir_exists(self) :
        if os.path.exists(self.outputDir) != True :
            try:
                os.makedirs(self.outputDir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    print('An exception on creating file')
                    raise

    def get_outputDir(self):
        return self.outputDir

    def get_inputDir(self):
        return self.inputDir
    
    
class filelist():
    def __init__(self,path):
        self.path = path
        
        for root, dir, files in os.walk(self.path):
            self.files = [f for f in files if not f.startswith('~') and f!='Thumbs.db']

        print(self.files)
        
    def get(self):
        return self.files    
    
