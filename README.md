# RC_prod  
С помощью данной MVP(минимально жизнеспособный продукт) демонстрируется работа модели рекомендательной системы(РС) интернет-магазина.  

## Бизнес-задача: увеличение продаж на 20% за счет использования РС.  
В прилагаемой [презентации](https://github.com/PavelNovikov888/RC_prod/blob/master/RS_presentation.odp) приведены данные проведенного [исследования](https://github.com/PavelNovikov888/RC_prod/blob/master/RS_research_kaggle.ipynb) и обоснованы причины выбора модели на основе библиотеки LightFM  
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

### Инструкция по использованию программы:
1. После появления на экране надписи:     
> * Serving Flask app 'houseapp.py'  
> * Debug mode: off  
> WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.   
> * Running on http://127.0.0.1:5000

2. Перейти в браузере по адресу [http://127.0.0.1:5000](http://127.0.0.1:5000)
3. Вы попадете в [меню с 2 кнопками](https://github.com/PavelNovikov888/RC_prod/blob/master/visual/choice.png). На этом этапе нужно сделать выбор - для пользователя или продукта нужны рекомендации
4. При выборе пользователя нужно ввести в появившемся поле произвольный номер пользователя.
> Алгоритм определит есть ли данные об этом пользователе среди предыдущих взаимодействий или нет. Если такие данные есть, то на экран выведутся [рекомендации](https://github.com/PavelNovikov888/RC_prod/blob/master/visual/user_hot.png) для двух случаев.

> Первый случай, это когда разрешены повторные покупки. К примеру, пользователь ранее купил молоко, и ему разумеется можно предложить его приобрести снова. Другой случай, если речь идет о товарах длительного пользования. К примеру, если пользователь ранее приобрел диван, то логично будет ему предложить купить не ещё один диван, а к примеру средства по уходу или защитный чехол. В нашей задаче мы не имеем возможности определять о каких именно товарах идет речь и о политике условного предприятия. Поэтому производится расчет для обоих случаев. Для наглядной демонстрации, того как работает программа для товаров длительного пользования рекомендую ввести в качестве userid *7575*.

> В случае, если это новый пользователь, для выбора рекомендаций будут использоваться признаки пользователя. Т.к. данных признаков у нас нет, то программа сгенерирует их из наиболее популярных признаков среди пользователей, сделавших покупки и на экран будут выведены [рекомендации](https://github.com/PavelNovikov888/RC_prod/blob/master/visual/user_cold.png) для нового пользователя. 

 5. При выборе продукта(item) нужно ввести в появившемся поле произвольный номер продукта.
> Алгоритм определит, были ли взаимодействия этого товара с пользователями ранее. Если да, как например в случае itemid *4041*, то на экран выведутся [рекомендации](https://github.com/PavelNovikov888/RC_prod/blob/master/visual/item_hot.png) для товаров имеющих историю взаимодействия с пользователями.

> Для нового продукта добавляется таблица схожести данного продукта с другими. Выбор [рекомендаций](https://github.com/PavelNovikov888/RC_prod/blob/master/visual/item_cold.png) товаров связан с наивысшим местом среди других предлагаемых пользователям товаров.

### Реализация принципов REST API  
Таблица маршрутизации   
| Тип запроса | routes | Параметры | Пояснение |
| ------------|---------|----------|----------------|
| GET | / |-| Запрос направляем на стартовую страницу выбора объектов выдачи рекомендаций|  
| POST | /submit |-| Перенаправляет запрос на страницу ввода id пользователя или товара  |
| GET | /users |-| Запрос направляет на страницу ввода userid  | 
| POST | /users |int| Запрос направляет на страницу с выданными рекомендациями|  
| GET | /items |-| Запрос направляет на страницу ввода itemid  |  
| POST | /items |int| Запрос направляет на страницу с выданными рекомендациями |  
|||| |     


