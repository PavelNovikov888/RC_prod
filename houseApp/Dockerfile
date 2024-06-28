FROM python:3.8-slim
WORKDIR /usr
COPY . .
RUN apt-get update && apt-get install -y gcc 
RUN pip install -r requirements.txt
ENV FLASK_APP=houseapp.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000
# ENTRYPOINT ["python"]
# RUN flask run
CMD [ "flask", "run" ]
# CMD ["./houseapp.py" ]
# /home/user/Prod_f/houseApp
