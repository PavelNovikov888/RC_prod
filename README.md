# RC_prod  
С помощью данной MVP(минимально жизеспособный продукт) демонстрируется работа модели рекомендательной системы(РС) интернет-магазина.  

## Бизнес-задача: увеличение продаж на 20% за счет использования РС.  
В прилагаемой [демонстрации]() приведены данные проведенного [исследования]() и обоснованы причины выбора модели на основе библиотеки LightFM  
MVP создана на основе Flask и принципов REST API

## MVP позволяет: 
- Рекомендовать items для users имеющих и не имеющих истории взаимодействия с items
- Рекомендовать только новые items или с допущением повторных покупок
- Рекомендовать users для items уже имеющихся в истории продаж и для вновь вводимых в ассортимент  
- находить похожие по свойствам items для более точной классификации и таргета

## Установка программы    
### 1 вариант: С помощью GitHub  
1. Клонировать данный репозиторий с помощью команды:    
   ` git clone https://github.com/PavelNovikov888/RC_prod.git `   
2. В терминале перейти в корневую папку данного проекта и выполнить команду:     
   `bash engage`  

### 2 вариант: С помощью DockerHub
> Docker образ программы хранится в DockerHub по адресу [https://hub.docker.com/r/pavelnovikov/reccom_image](https://hub.docker.com/r/pavelnovikov/reccom_image)

1. Загрузить в рабочую папку Docker образ(image) командой:  
   `docker pull pavelnovikov/reccom_image:1.0`
2. Проверить появился ли данный образ в Docker:  
   `docker images`
3. Запустить контейнер командой:  
   `docker run -p 5000:5000 reccom_image:1.0`  
    
4. После появления на экране надписи:     
> * Serving Flask app 'houseapp.py'  
> * Debug mode: off  
> WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.   
> * Running on http://127.0.0.1:5000

5. Перейти в браузере по адресу [http://127.0.0.1:5000]()
   
  Маршрутизация:
