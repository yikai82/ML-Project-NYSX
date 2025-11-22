<!-- <p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p> -->

<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange <b>Demo</b><br>
  </h1>

<p align="center"> 
  <a href="/experiments/Readme.md">Experiments Play Zone 🛝 </a> •
  <a href="/production/Readme.md">Production ⛑️ 🏭</a> •
  <a href="/README.md">Main Page 🏠
</a><br> 
</p>


Hello, 

Welcome to the **Demo** zone! We will demonstrate how to use the docker + MLflow for a single AAPL stock prediction and publish the result as an interactive webpage   

## Update log:  
- 2025-11-22: initial version



## Steps 

1. Make sure you have [Docker and the environment](/README.md) set up properly. I will use the `dsi_participant` environment as an example here. Feel free to use any appropriate environment.

2. Start Docker and activate your environment in the terminal:   

```bash
cd /path/to/ML3-Team-Project/demo
conda activate dsi_participant  # activate environment 
docker compose -f docker-compose-demo.yml up -d compose up docker 
docker ps # should show the list of running containers.
```  
3. If everything passes, you should be able to see the terminal out like [this](/demo/images/demo_docker.png).

4. Run the following to test Docker + MLflow

```bash
python test.py # this test mlflow.log_param and mlflow.log_metric functions 
python test_mlflow.py # test mlflow with a simple logistic regression 
```
&emsp; If both are passed, you should be able to open link (http://localhost:5002/#/experiments/0/) and see something like this: 
<br>   
<img src="images/demo_mlflow.png" width="1200" align="left" style="margin-left: 20px; margin-bottom: 40px;">

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
5. Inspect `demo_LSTM_v04.3.py` and Run it

    - Open `demo_LSTM_v04.3.py` in any command line editor or VScode and inspect all the paths if you want to edit anything: 
        - exp_name = "NYSX_LSTM_demo"
        - output_PATH = "./output"
        - output_PATH = "./output", (inside the for loop)
        - model_PATH = "./output"
        - RUN_NAME = f"TEST_{Tick}_LSTM_ver04.3"
        - FILE_PATH = "/../data/raw"
        - fig.write_html(f"{output_PATH}/{Tick}_filename.html")
        - fig.write_image(f"{output_PATH}/{Tick}_filename.html")
        - model_lstm.save(f"{output_PATH}/{Tick}_lstm_model.keras")

    👉 Highly recommend to use Ctrl + F and search keyworld seach as "save", "write".etc


```bash
python demo_LSTM_v04.3.py
```

<br>   
<img src="images/demo_AAPL.png" width="1200" align="left" style="margin-left: 0 px; margin-bottom: 40px;">