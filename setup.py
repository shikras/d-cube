import setuptools

setuptools.setup(
    name='ddd-dataset',
    version='0.1.1',
    author='Chi Xie',
    author_email='chixie.personal@gmail.com',
    description='Toolkit for Description Detection Dataset ($D^3$)',
    long_description='Toolkit for Description Detection Dataset ($D^3$): A detection dataset with class names characterized by intricate and flexible expressions, for the Described Object Detection (DOD) task.',
    long_description_content_type='text/markdown',
    license='CC BY-NC 4.0',
    packages=['d_cube'],
    package_dir={"d_cube": "d_cube"},
    url='https://github.com/shikras/d-cube',
    project_urls={
        "Bug Tracker": "https://github.com/shikras/d-cube/issues",
    },
    install_requires=['numpy', 'pycocotools', 'opencv-python', 'matplotlib'],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
    ],
)
