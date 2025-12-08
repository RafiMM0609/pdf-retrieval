# FROM python:3.10

# ENV PYTHONUNBUFFERED=1
# # WORKDIR /app
# WORKDIR /app-knowledge


# RUN apt-get update \
# && apt-get -y upgrade \
# && apt-get -y install ghostscript \
# && apt-get -y install qpdf \
# && apt-get -y install tesseract-ocr \
# && apt-get -y install git autotools-dev automake libtool libleptonica-dev \
# && apt-get -y install pngquant \
# && apt-get -y install unpaper \
# && apt-get -y install ffmpeg libsm6 libxext6 \
# && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/agl/jbig2enc \
# && cd jbig2enc \
# && ./autogen.sh \
# && ./configure && make \
# && make install \
# && cd .. \
# && rm -rf jbig2enc

# RUN pip install --upgrade pip

# RUN pip install git+https://github.com/ocrmypdf/OCRmyPDF.git@3a75b2074092d9e23036ec7db72c07417682afe3


# RUN pip install sentence-transformers

# COPY requirements.txt . 

# RUN pip install -r requirements.txt

FROM ectraction-pdf-pdf-rag-api

# =====================HERE IS NEW==========================================

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
