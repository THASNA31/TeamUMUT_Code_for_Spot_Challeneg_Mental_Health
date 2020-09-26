# TeamUMUT Code for Spot Challenge in Mental Health issues during COVID-19
This repository contains all the code, data, summary, and executable files created for the "Code for spot " challenge on mental health issue developed by team UMUT.

Some pointer:
1. The "dist" folder contains the executable files.
2. The ".ipynb" extension files contains the jupyter scripts for each tool.

Summary:

Hispanics population in the US are disproportionately affected by COVID-19, especially in mental health issues. A survey by the Pew Research Center found that 28% of Hispanics experiencing “high” psychological distress. According to the CDC, on average, the percentage of Hispanic population reporting symptoms of depressive disorder is 2.8% higher than the average population, 6% and 8% higher than White and Asian communities, respectively.

To deal with this issue, we have developed a Decision Support System (or DSS) which can be used by local agencies, such as, Give an Hour. The DSS consists of two tools: (1) predictive tool and (2) Markov Decision Process (MDP) tool. The tools are in executable format. The predictive tool forecasts the severity of depressive and anxiety disorder of our target user group at each state for the next period by considering several predictors. The MDP tool assists the local agencies in making better-informed actions based on the severity level of anxiety and depressive disorders for the user group across the states. 

The predictive tool contains a linear-regression based model to predict the level of response variable (RV), percentage of Hispanic population showing symptoms of anxiety and depressive disorders, at each state in the US for the next 2 weeks. We have used the following indicators as the predictors for Hispanics: (i) COVID-19 related deaths, (ii) percentage population without health insurance, (iii) total unemployment claims. We have found 'US State' to be an insignificant predictor, which led to developing predictive models for each state separately. The executable creates visualizations in html format for R-squared, RMSE, and the predicted response for next period.

The MDP tool consists of a set of states, actions, transition probability matrix (TPM), and rewards. We consider three states for our MDP model: Red-alarming, Yellow-severe, and Blue-moderate. Two thresholds values on the RV selected by the decision maker (DM) will determine the MDP state for each state in the US. Based on a preliminary literature review, we have defined the following actions: (i) hire a celebrity (HC), (ii) increase budget in social media advertising by 1% (SMA), (iii) increase paid positions in telemedicine by 1% (TM), (iv) create one art therapy (AT), (v) increase social awareness activity by 1% (SAA), (vi) increase number of support groups by 1% (SG). Notice, we have not found any specific mental health solution which is customized for Hispanics only. We consider a “cost-ordering” approach to develop the TPM for the MDP model. Each action is assumed to have a cost associated with them and the action with the highest cost has the highest probability to turn a Red state to Blue.

We believe, the inclusion of a prescriptive tool as the MDP tool makes our product unique. While basic data-science methodologies can do wonders with predictive formulations, an MDP tool can help a DM to take time-appropriate action. Executables for both tools can be implemented right now. While, a short-term collaboration between our team and the agencies can really improve the usability of this tool drastically.
