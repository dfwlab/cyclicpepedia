FROM silverlogic/python3.8
MAINTAINER DingfengWu dfw_bioinfo@126.com

USER root

RUN apt update && \
	apt install --yes r-base &&\
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install rdkit==2022.9.3 \
	Django==3.2.16 \
	django-haystack==3.1.1 \
	django_import_export==3.0.2 \
	drf_haystack==1.8.11 \
	whoosh==2.7.4  \
	biopython==1.81 \
	scipy==1.10.1 \
	networkx==3.1 \
	matplotlib==3.7.4 \
	pandas==2.0.3 \
	ipython==8.12.3 \
	django-tables2==2.7.0 \
	rpy2==3.5.15

# install r packages
RUN R -e "install.packages(c('Peptides'), dependencies=TRUE, repos='http://cran.rstudio.com/')" &&\
    rm -rf /tmp/*

WORKDIR /Cyclicpepedia

CMD ["python3", "manage.py", "runserver", "0.0.0.0:8000"]





