<p align="center">
  <img src="[insert XXX IMAGE URL]" alt="Image Place" width="120">
</p>

#
<h1 align="center">
  Yi-Kai's AI/ML New York Stock Exchange Play Zone 🛝 <br>
  </h1>

<!-- <p align="center">
  Enter Text Here <br>
  <a href="#key-note-and-important-concept">Key Notes</a> 
  <a href="URL">Enter Text Here</a> •
  <a href="">Resouces</a><br>
</p> -->

## Rationale and Recent Update 

### Question to Answer: 

`**Can we oredict the next day price by giving a specific input using lean LSTM algorithms?**` 

### 🧠 Rationale: 

Why LSTM? Save time and resource for feature engineering. 

- "Most of the business problems are not ML problems, and most of the ML problems are not business problems" from Production Lecture Day 1 : 36:58.
  - Optimizing an ML model is not the same thing as optimizing a business problem. 

- ML system: Learn automatically and gain from the experience (*the experience: in plain English, more data, more observation*); however, more data or experience does not necessarily guarantee better performance.

- **The 80/20 rule**: Roughly 80% of outcomes result from 20% of causes or inputs. Therefore, to have the model predict with a reasonable confidence level, we only need to capture the 20% the most important information or patterns.   



### 🧭 🗺️ Current Roadmap:

This idea here is focuses on isolating core predictive patterns rather than modeling every variable. The goal is to extract essential signals using available and ready to use algorithms by adopting the 80/20 rules — to keep the model both interpretable and computationally efficient.

  1. Quick test of available common stock models (ARIMA, LSTM, etc.) with representive stocks (AAPL, INTC, MSFT) ✅  

  2. Assess the baseline performance using randomness robustness, regression, residual analysis (Test 1 to Test 6)✅  
  
  3. Tuning by featuring engineering (`close_price`, `return_7d`,`streak_up`, `streak_down`,`range_ratio`) and re-assess the tuned model for randomness robustness (Test 7, 8, 9)✅  
  
  4. Applying the same approach to the same sectors"TBD/exploratory?(TBD)”
  
      - Selecting indicator stocks (TBD)  

      - Exploring AutoML for automation for tuning on hyperparameters (TBD)

  5. UMAP-based analysis for poor performers (TBD) 
   .....  
   .....  
  X. Scale-up to other sectors?(TBD)    



### Update log: 
- Date: 2025-11-11, update and clean up test 1 to 9.   
- Date: 2025-11-09, upload test 4 to 9, Readme.md 
- Date: 2025-11-06, upload test 1 to test 3

## Content  
* [Experiment Note](#experiment-note)   
* [1. 🏛️ Libraries 🏛️]()  
* [2. ⚔️🛡️ PLAY BOOK](#)
  * [2.1]
  * [2.2]  
* [3. Resources](#resources) 



---  
## Experiment Note:
> [!Note]  
> 1. Test 1 to Test 3: quick test of ARIMA and LSTM using AAPL, INTC, and MSFT.
> 
> 2. Test 4: Further deep dive into ARIMA, no good result ❎.
> 
> 3. **Test 5**: Further deep dive into LSTM, interesting results. Testing model randomness robustness (**same random seed x 3 runs**), testing the concept of prediction of the next day closing price using the last 20-day closing price.  
> 
> 4. **Test 6**: Further deep dive into LSTM, interesting results. Testing model randomness robustness (**three different random seed x 3 runs**), testing the concept of prediction of the next day closing price using the last 20-day closing price. 
> 
> 5. **Test 7 to Test 9**: Adding extra **feature engineering** (6 features total) to increase the model stability and testing the robustness with APPL, INTC, and MSTF.

<!-- > [!mportantce]  
>  
> 
>   
 -->


<!-- 
> [!WARNING]  
> 1. 
> -->


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 1. 🏛️ Libraries 🏛️

```python



```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 

---
## 2. ⚔️🛡️ PLAY BOOK

### 2.1 Concept 

```text

    ┌─────────────────────────────────────────┐
    │               Input Layer               │
    │         (20 timesteps × 6 features)     │
    └─────────────────────┬───────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │                LSTM Layer               │
    │              50 hidden units            │
    │      Learns temporal dependencies       │
    │                                         │
    │           Total params: 36,755          │
    │           Trainable params: 12,251      │
    │           Non-trainable params: 0       │
    │           Optimizer params: 24,504      │
    └─────────────────────┬───────────────────┘
                          │
                          ▼
    ┌─────────────────────────────────────────┐
    │               Dense Output              │
    │                 (1 neuron)              │
    │           Predicts next-day price       │
    └─────────────────────────────────────────┘

```






```python
# Enter some code here

```
<br>

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
### 2.2  

```python
# Enter code here

```

<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
### 2.3 


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub> 


---
## 3. Function List  



[Reference for scoring](https://scikit-learn.org/stable/modules/model_evaluation.html?utm_source=chatgpt.com)




<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## 3. Extra


<sub>[↥ back to top](#content)&emsp;|&emsp;[Return Main Page 🏠](/README.md) </sub>  

---
## Resources
1. [W3School Pandas - DataFrame Reference](https://www.w3schools.com/python/pandas/pandas_ref_dataframe.asp)