from setuptools import setup
setup(
  name = 'calibration',
  packages = ['calibration'],
  version = '1.0',
  description = 'Performs GP calibration on network of colocated sensors.',
  author = 'Mike Smith',
  author_email = 'm.t.smith@sheffield.ac.uk',
  url = 'https://github.com/lionfish0/calibration.git',
  download_url = 'https://github.com/lionfish0/calibration.git',
  keywords = ['calibration','sensors','internet of things','iot','network','pollution','gaussian process','variational inference','tensorflow'],
  classifiers = [],
  install_requires=['numpy<1.19.0,>=1.16.0','gast==0.3.3','absl-py','tensorflow-probability','pandas','scipy','gpflow'],
)
