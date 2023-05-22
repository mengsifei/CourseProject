# Video analysis to get the best face frame from video
## Link to web-app
http://1462839-cj10416.tw1.ru/
## Documentation for users who want to use the web-app.
This web-application was developed by Sifei Meng, a third-year student from the Higher School of Economics. The primary objective of this application is to generate the optimal frame that encompasses a face from a video. The user can upload a video that is less than 60MB on the home page. On the subsequent row, the user can regulate several parameters, which are described in detail below.

The videos have to be in formats 'mp4', 'mov', 'wmv', 'avi', 'mkv'.

* **NumFrame** denotes the number of frames that the program will choose from. By default, the program will scrutinize the entire video. The value must be non-negative; if it is negative, the algorithm will take the absolute value. If the number is zero, the program will examine all the frames with corresponding fps. If there are no suitable frames within the given number of frames, and the algorithm has not examined the entire video, it will automatically continue analyzing the subsequent NumFrame frames until the video concludes. 
* **FPS** refers to the number of frames that will be analyzed per second. By default, the program will analyze three frames per second, but the user can customize this value. The value must be non-negative; if it is negative, the program will use the absolute value. If the user sets the value greater than the fps of the original video, then the program will scrutinize every frame of each second. 
* **Frame_size** enables the user to resize the width of the optimal frame. The height of the frame will be adjusted proportionally.
* **Interrupt** allows the user to halt the process by clicking on the "Interrupt" button displayed on the page while the algorithm is running. The process will cease within one second. 
* **Download** file allows the user to effortlessly download the file by clicking on the "Download file" button after generating the best frame. The name of the jpg file will be "best_frame_XXXXXXXX.jpg," where X represents random file names. 
* **More information about session** provides insight into how long the session lasted. This is useful for comparing the productivity of the web-app with other analogous programs.
* **Try again** redirects the user to the home page. 

At the bottom of each page, four buttons are available. Code on Github is a link to the repository on Github. About this project is the documentation of the project. Leave feedback is a survey where the user can submit feedback. Home page enables the user to return to the home page at any time. 

## User Guide for Local Installation of Web-Application

1. Repository Cloning: The first step is to clone the repository from Github to your local machine using Git. This can be done by executing the following command in your terminal:

```git clone https://github.com/mengsifei/CourseProject.git```

2. Dependency Installation: Once the repository has been cloned, the user must verify the package versions and install the required dependencies by running the following command:

```. ./bootstrap.sh```

3. Server Startup: After installing the dependencies, the server can be started by executing the following command:

```. ./start.sh```

## Документация для пользователей, желающих использовать веб-приложение.

Данное веб-приложение было разработано Мэн Сыфэй, студенткой третьего курса Высшей школы экономики. Основная цель данного приложения заключается в получении лучшего изображения лица видеозаписи. Пользователь может загрузить видео размером менее 60 МБ на домашнюю страницу. На следующей строке пользователь может регулировать несколько параметров, которые подробно описаны ниже.

Видео должны быть в форматах 'mp4', 'mov', 'wmv', 'avi', 'mkv'.

* **NumFrame** определяет количество первых кадров, которые будут рассматриваться программой. По умолчанию программа будет анализировать всё видео. Значение должно быть неотрицательным; если оно отрицательное, алгоритм возьмет абсолютное значение. Если число равно нулю, программа будет анализировать все кадры с соответствующим fps. Если в данном количестве кадров нет подходящих, и алгоритм не проанализировал всё видео, он автоматически продолжит анализировать следующие NumFrame кадров до тех пор, пока не получено изображение, соответствующее критериям.

* **FPS** относится к количеству кадров, которые будут анализироваться в секунду. По умолчанию программа будет анализировать три кадра в секунду, но пользователь может настроить это значение. Значение должно быть неотрицательным; если оно отрицательное, программа будет использовать абсолютное значение. Если пользователь установит значение больше fps исходного видео, то программа будет анализировать каждый кадр каждой секунды.

* **Frame_size** позволяет пользователю изменять ширину оптимального кадра. Высота кадра будет соответственно скорректирована.

* **Interrupt** позволяет пользователю остановить процесс, нажав кнопку "Прервать", отображаемую на странице во время работы алгоритма. Процесс прекратится в течение одной секунды.

* **Download file** позволяет пользователю легко загрузить файл, нажав кнопку "Загрузить файл" после создания лучшего кадра. Имя jpg-файла будет "best_frame_XXXXXXXX.jpg", где X обозначает случайные символы.

* **“More information about session”** предоставляет представление о том, сколько времени длилась сессия. Это полезно для сравнения производительности веб-приложения с другими аналогичными программами.

* **Try again** перенаправляет пользователя на домашнюю страницу.

Внизу каждой страницы доступны четыре кнопки. Code on Github - это ссылка на репозиторий на Github. About this project - это документация проекта. Leave feedback - это опрос, в котором пользователь может оставить отзыв. Home page позволяет пользователю вернуться на домашнюю страницу в любое время.

## Документация для локальной установки веб-приложения

1. Клонирование репозитория: Первый шаг - склонировать репозиторий с Github на ваше локальное устройство, используя Git. Это можно сделать, выполнив следующую команду в терминале:

```git clone https://github.com/mengsifei/CourseProject.git```

2. Установка зависимостей: После того, как репозиторий был склонирован, пользователь должен проверить версии пакетов и установить необходимые зависимости, выполнив следующую команду:

```. ./bootstrap.sh```

3. Запуск сервера: После установки зависимостей сервер может быть запущен, выполнением следующей команды:

```. ./start.sh```