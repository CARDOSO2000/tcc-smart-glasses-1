# Deploy da IA tcc na Azure

1° passo criar o DockerFile para gerar uma docker e rodar o código da ia como a ia foi feita para escutar audios e classificar eles como não havia entrada de audio par o container foi colocado junto a docker um arquivo de som mockado:

![image](https://user-images.githubusercontent.com/56920123/204935073-285f4e14-388e-45a6-9fb6-9e4b1554aff5.png)

2° Após a Docker criada foi testado se o codigo estava rodando no container local:
![image](https://user-images.githubusercontent.com/56920123/204935494-b2fafcfd-052a-44d0-b2d1-c0ea77746c04.png)

3° Após tudo verificado foi feito o deploy do container na Azure:
![Captura de tela 2022-11-30 211005](https://user-images.githubusercontent.com/56920123/204936239-ae72be81-3071-4427-9793-05878aa62ab3.png)
![Captura de tela 2022-11-30 211029](https://user-images.githubusercontent.com/56920123/204936249-7cca5ab1-287b-44c0-a22e-34d373f80253.png)

4° Deploy feito com sucesso:
![image](https://user-images.githubusercontent.com/56920123/204936369-92e452ca-34e7-422b-8e73-f99bff7574e7.png)

5° Logs do container em execução:
![image](https://user-images.githubusercontent.com/56920123/204937051-bb0d5b9d-5621-4072-8245-f1d5d77ad093.png)
![image](https://user-images.githubusercontent.com/56920123/204937099-854ddcc9-374e-42f1-a398-a5d95de300e4.png)


