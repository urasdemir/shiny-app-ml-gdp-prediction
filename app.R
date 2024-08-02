# Load necessary libraries
library(WDI)
library(dplyr)
library(caret)
library(shiny)
library(e1071) # For SVM
library(ggplot2)
library(glmnet) # For Ridge and Lasso
library(countrycode)
library(gbm)
library(kernlab)

# Define function to fetch and preprocess data
fetch_data <- function() {
  indicators <- c("NY.GDP.PCAP.CD", "SE.TER.ENRR", "SL.UEM.TOTL.ZS", "FP.CPI.TOTL.ZG", "SP.POP.GROW", "NE.CON.GOVT.ZS")
  start_year <- 2000
  end_year <- 2023
  data <- WDI(indicator = indicators, start = start_year, end = end_year)
  colnames(data) <- c("country", "iso2c", "iso3c", "year", "GDP_per_capita", "Tertiary_Enroll", "Unemployment", "Inflation", "Population_Growth", "Gov_Expenditure")
  data$country_type <- countrycode(data$iso2c, "iso2c", "country.name")
  data <- data[!is.na(data$country_type), ]
  data <- na.omit(data)
  data$Tertiary_Enroll <- pmin(pmax(data$Tertiary_Enroll, 0), 100)
  data$Unemployment <- pmin(pmax(data$Unemployment, 0), 30)
  data$Population_Growth <- pmin(pmax(data$Population_Growth, 0), 25)
  data$Gov_Expenditure <- pmin(pmax(data$Gov_Expenditure, 0), 50)
  data <- data %>% mutate(logGDP_per_capita = log(GDP_per_capita))
  return(data)
}

# Fetch data
data <- fetch_data()

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$logGDP_per_capita, p = 0.8, list = FALSE, times = 1)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Normalize the data
preProc <- preProcess(trainData[, c("Tertiary_Enroll", "Unemployment", "Inflation", "Population_Growth", "Gov_Expenditure")], method = c("center", "scale"))
trainDataNorm <- predict(preProc, trainData)
testDataNorm <- predict(preProc, testData)

# Train models
models <- list()
# Linear Regression Model
models$lm <- train(logGDP_per_capita ~ Tertiary_Enroll + Unemployment + Inflation + Population_Growth + Gov_Expenditure, data = trainDataNorm, method = "lm",
                   trControl = trainControl(method = "cv", number = 5))
# Ridge Regression Model
models$ridge <- train(logGDP_per_capita ~ Tertiary_Enroll + Unemployment + Inflation + Population_Growth + Gov_Expenditure, data = trainDataNorm, method = "glmnet",
                      trControl = trainControl(method = "cv", number = 5), tuneGrid = expand.grid(alpha = 0, lambda = 10^seq(10, -2, length = 100)))
# GBM Model
gbmGrid <- expand.grid(interaction.depth = c(1, 5, 9), n.trees = (1:30) * 50, shrinkage = c(0.01, 0.1), n.minobsinnode = 20)
models$gbm <- train(logGDP_per_capita ~ Tertiary_Enroll + Unemployment + Inflation + Population_Growth + Gov_Expenditure, data = trainDataNorm, method = "gbm",
                    trControl = trainControl(method = "cv", number = 5), verbose = FALSE, tuneGrid = gbmGrid)
# SVM Model
models$svm <- train(logGDP_per_capita ~ Tertiary_Enroll + Unemployment + Inflation + Population_Growth + Gov_Expenditure, data = trainDataNorm, method = "svmRadial",
                    trControl = trainControl(method = "cv", number = 5), tuneLength = 10)

# Define UI
ui <- fluidPage(
  titlePanel("GDP Prediction with Supervised Learning"),
  sidebarLayout(
    sidebarPanel(
      h3("Instructions"),
      p("This app predicts log GDP per capita based on education, unemployment, inflation rates, population growth, and government expenditure using various models trained on World Bank data."),
      p("Select the model you want to use, adjust the sliders to set the tertiary education enrollment rate, unemployment rate, inflation rate, population growth rate, and government expenditure rate, then click 'Predict log GDP per Capita' to see the prediction."),
      radioButtons("model", "Choose Model:", choices = c("Linear Regression" = "lm", "Ridge Regression" = "ridge", "Gradient Boosting Machine" = "gbm", "Support Vector Machine" = "svm"), selected = "lm"),
      sliderInput("tertiary_enroll", "Tertiary Enrollment Rate (%):", min = 0, max = 100, value = mean(data$Tertiary_Enroll), step = 0.1),
      sliderInput("unemployment", "Unemployment Rate (%):", min = 0, max = 30, value = mean(data$Unemployment), step = 0.1),
      sliderInput("inflation", "Inflation Rate (%):", min = 0, max = 100, value = mean(data$Inflation), step = 0.1),
      sliderInput("population_growth", "Population Growth Rate (%):", min = 0, max = 25, value = mean(data$Population_Growth), step = 0.1),
      sliderInput("gov_expenditure", "Government Expenditure (% of GDP):", min = 0, max = 50, value = mean(data$Gov_Expenditure), step = 0.1),
      actionButton("predict", "Predict log GDP per Capita"),
      actionButton("reset", "Reset")
    ),
    mainPanel(
      h3("Prediction Results"),
      textOutput("prediction"),
      textOutput("metrics"),
      h3("Log GDP per Capita Over Time"),
      plotOutput("plot"),
      h4("Explanation"),
      p("This graph shows how log GDP per capita has changed over time for various countries. The green points represent the training data used to build the model, and the red points represent the testing data used to evaluate the model. The blue line shows the overall trend based on the selected model."),
      h3("Predicted vs. Actual Log GDP per Capita"),
      plotOutput("plot2"),
      h4("Explanation"),
      textOutput("model_explanation"),
      br(),
      p("Shiny app by Uras Demir")
    )
  )
)

# Define Server Logic
server <- function(input, output, session) {
  prediction <- reactiveVal(NULL)
  metrics <- reactiveVal(NULL)
  predictions_made <- reactiveVal(FALSE)
  results_df <- reactiveVal(NULL)
  model_predictions <- reactiveVal(NULL)
  
  observeEvent(input$predict, {
    req(input$tertiary_enroll, input$unemployment, input$inflation, input$population_growth, input$gov_expenditure)
    newdata <- data.frame(Tertiary_Enroll = input$tertiary_enroll, Unemployment = input$unemployment, Inflation = input$inflation, Population_Growth = input$population_growth, Gov_Expenditure = input$gov_expenditure)
    newdataNorm <- predict(preProc, newdata)
    
    model <- models[[input$model]]
    predicted_value <- predict(model, newdataNorm)
    test_predictions <- predict(model, testDataNorm)
    
    mae <- mean(abs(test_predictions - testDataNorm$logGDP_per_capita))
    rmse <- sqrt(mean((test_predictions - testDataNorm$logGDP_per_capita)^2))
    
    prediction(predicted_value)
    metrics(paste("MAE:", round(mae, 2), "| RMSE:", round(rmse, 2)))
    
    # Create a new dataframe for storing predictions
    results <- data.frame(Actual = testDataNorm$logGDP_per_capita, Predicted = test_predictions, Year = testData$year)
    results_df(results)
    
    # Store model predictions for trend line in first graph
    model_pred <- data.frame(Year = data$year, Predicted = predict(model, predict(preProc, data)))
    model_predictions(model_pred)
    
    predictions_made(TRUE)
  })
  
  observeEvent(input$reset, {
    updateRadioButtons(session, "model", selected = "lm")
    updateSliderInput(session, "tertiary_enroll", value = mean(data$Tertiary_Enroll))
    updateSliderInput(session, "unemployment", value = mean(data$Unemployment))
    updateSliderInput(session, "inflation", value = mean(data$Inflation))
    updateSliderInput(session, "population_growth", value = mean(data$Population_Growth))
    updateSliderInput(session, "gov_expenditure", value = mean(data$Gov_Expenditure))
    prediction(NULL)
    metrics(NULL)
    results_df(NULL)
    model_predictions(NULL)
    predictions_made(FALSE)
  })
  
  output$prediction <- renderText({
    if (is.null(prediction())) {
      "Prediction will be displayed here."
    } else {
      pred_value <- round(prediction(), 2)
      paste("Predicted log GDP per Capita:", pred_value)
    }
  })
  
  output$metrics <- renderText({
    if (is.null(metrics())) {
      "Metrics will be displayed here."
    } else {
      metrics()
    }
  })
  
  output$plot <- renderPlot({
    req(predictions_made())
    ggplot(data, aes(x = year, y = logGDP_per_capita)) +
      geom_point(data = trainData, aes(x = year, y = logGDP_per_capita), color = "green") +
      geom_point(data = testData, aes(x = year, y = logGDP_per_capita), color = "red") +
      geom_smooth(data = model_predictions(), aes(x = Year, y = Predicted), method = "lm", se = FALSE, color = "blue") +
      labs(title = "Log GDP per Capita Over Time", x = "Year", y = "Log GDP per Capita", caption = "Green: Training data, Red: Testing data, Blue: Predicted values based on selected model") +
      theme_minimal()
  })
  
  output$plot2 <- renderPlot({
    req(predictions_made())
    results <- results_df()
    ggplot(results, aes(x = Actual, y = Predicted)) +
      geom_point(color = "blue") +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
      labs(title = "Predicted vs. Actual Log GDP per Capita", x = "Actual log GDP per Capita", y = "Predicted log GDP per Capita") +
      theme_minimal()
  })
  
  output$model_explanation <- renderText({
    switch(input$model,
           "lm" = "Linear Regression is a basic yet powerful statistical method for modeling the relationship between a dependent variable and one or more independent variables. It aims to find the best fit line through the data points. Linear regression is widely used due to its simplicity and interpretability. Interpretation: The closer the points are to the red dashed line, the better the model is at predicting the actual values.",
           "ridge" = "Ridge Regression is a type of linear regression that includes a regularization term (shrinkage parameter) to prevent overfitting. It is particularly useful when the data has multicollinearity (independent variables that are highly correlated). Ridge Regression helps to maintain model complexity without overfitting the data. Interpretation: The closer the points are to the red dashed line, the better the model handles multicollinearity and prevents overfitting.",
           "gbm" = "Gradient Boosting Machine (GBM) is an advanced machine learning technique that builds models sequentially. Each new model focuses on correcting errors made by the previous ones, leading to improved accuracy. GBM is particularly effective for handling complex datasets with many variables. It combines the strengths of multiple weak learners to form a strong predictive model. Interpretation: The accuracy of the model is indicated by how close the points are to the red dashed line, showing the sequential improvement.",
           "svm" = "Support Vector Machine (SVM) is a powerful algorithm used for classification and regression tasks. It works by finding the hyperplane that best separates data points into different classes. For regression, SVM aims to find the best fit line or curve that predicts the target variable accurately. SVM is particularly effective in high-dimensional spaces and when the number of dimensions exceeds the number of samples. Interpretation: The proximity of the points to the red dashed line indicates the accuracy of the SVM in predicting log GDP per capita."
    )
  })
}

# Run the app
shinyApp(ui = ui, server = server)
