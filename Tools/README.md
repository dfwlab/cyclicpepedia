# Algorithms of CyclicPepedia

Here is the code for the cyclic peptide tool used by CyclicPepedia and its test notebooks. You need to ensure that the program is valid in the same environment (build a Docker image through <code>../dockerfile</code> and <code>pip install jupyterlab</code>), or pull docker image through this link: <code>docker pull registry.cn-hangzhou.aliyuncs.com/dfwlab/cyclicpepedia:20240106</code>. You can use this image with the following command like: <code>docker run -it -p 80:8888 -v $(PWD):/Cyclicpepedia $imageName:$tag /bin/bash</code>.

* 1. Struc2Seq (structure2sequence.py) and Test_Struc2seq.ipynb
* 2. Seq2Struc (sequence2structure.py) and Test_Seq2struc.ipynb
* 3. ga (graph_alignment.py) and Test_GraphAlignment.ipynb
* 4. pp (peptide_properties.py) and Test_Peptide_properties.ipynb
* 5. RDKit_format_issue.ipynb

