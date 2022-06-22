import setuptools

with open("README.md", "r", encoding="utf-8") as s:
    long_description = s.read()

# ---------- Check if opencv-contrib is already installed ------------
opencv_installed = False
try:
    import pkg_resources
    import re
    installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
    for pkg, ver in installed_packages.items():
        matched = re.match("opencv-contrib", pkg, re.IGNORECASE)
        if matched:
            min_ver = 4.5
            max_ver = 5.0
            this_ver = int(ver.split(".")[0])
            if min_ver < this_ver < max_ver:
                opencv_installed = True
            else:
                opencv_installed = False
except Exception as e:
    print("There was an exception, but I skipped it\n", e)


package_list = [
    "numpy>=1.0",
    "tifffile>=2020.0",
    "pandas>=1.0",
    "dask>=2020.0",
    "scikit-learn>=1.0",
    "scikit-image>=0.19",
    "jsonschema>=4.0"
]
if not opencv_installed:
    package_list.append("opencv-contrib-python>=4.5,<5.0")

# -------------------

setuptools.setup(
    name="lia",
    version="1.0",
    packages=["lia"],
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    license="GPLv3",
    author="Vasyl Vaskivskyi",
    author_email="vaskivskyi.v@gmail.com",
    description="LIA: Large Image Aligner for microscopy images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
    install_requires=package_list,
    include_package_data=True,
    data_files=[("lia", ["shared_modules/config_schema.json"])],
    entry_points={
        "console_scripts": [
            "lia = lia.__main__:main"
        ]
    }
)
