<!-- insert image  -->
<!-- <p align="center">
  <img src="" alt="Image Plance" width="180">
</p> -->

<p>
<h1 align="center">
  Capstone Project: The Next Day Stock Price Prediction 
  </h1>

> [!IMPORTANT]  
> <i><h3 align="center">Most of the business problems are not ML problems, and most of the ML problems are not business problems. Optimizing an ML model is not the same thing as optimizing a solution for a business problem </h3><p>  
> <h4 align="right"> - from Production Lecture Day 1 : 36:58</h4><p>
>
> <h3 align="center">A Machine Learning System is a system that can learn automatically to improve its performance</h3></i>    
>  
> This capstone project showcases what I learned during a 16-week intensive AI/ML course offered by the University of Toronto’s Data Science Institute. I am not a financial professional, but I do invest in the market as a side pursuit, chasing the occasional moonshot 🌛 🏹.

---

<p align="center">
  <a href="#key-note-and-important-concept">Key Notes</a> •
  <a href="URL">Fly to Course Recording</a> •
  <a href="#resources">Resouces</a><br>
  <br>
  **Enter Text<br>
</p>


---
LSTM = Long Short-Term Memory | ML = Machine Learning 

## Business Problem: 

<table>
<tr>
<td>

Our client recently experienced market losses triggered by a high-profile tweet. To improve their ability to respond to sudden shifts, the client wants to explore LSTM-based stock prediction models. Instead of maintaining one model per stock, they’re interested in whether a sector-level model could learn shared patterns and then be applied to individual tickers.

**Constraint**: Maximum training epochs = 20 to enable fast iteration and leave room for future feature additions.

</td>

<td style="width: 150px; padding-left: 20px;">
  <img src="images/trump tweetjpg" width="500">
</td>
</tr>
</table>

**Project goal**: Build a simple LSTM models to forecast next-day stock prices, compare their performance, and evaluate whether a single sector-level model can match the performance of stock-specific models.

---

## 2. Repo layout

```text





```
---

## 3. Environment setup (Git Bash / Windows)

```bash
# create & activate venv in Git Bash
python -m venv .venv
source .venv/Scripts/activate

# install minimal packages
pip install -r requirements.txt
