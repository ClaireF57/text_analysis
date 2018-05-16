########################Advanced numerical Methods and Data Analysis##############################
## Job Scraper
# Purpose: Scraping jobs, tf.idf get list of frequent skills
# Last change: 13.05.2018
######################################################################################################


#load packages setting static variables
library(rvest)
library(httr)
library(dplyr)
library(jsonlite)
library(data.table)
#fixed vars
wd <- setwd('D:/HSG/FS18/Numerical_Methods/TextAnalysisML')
wd <- setwd('D:/HSG/FS18/Numerical_Methods/TextAnalysisML')


# Setting up Scraper ------------------------------------------------------
scrapper <- function(jobTitle, location,maxnumber){
  # define base url
  
  BASE_URL <- 'https://www.indeed.com/jobs?q='
  
  
  # construct search url
  jobTitle <- gsub(" ", "+", jobTitle)
  location <- gsub(" ","+",location)
  pages <- seq(0, (maxnumber-10), by=10)
  titles <- vector('character')
  desc <- vector('character')
  id <- vector('character')
  for (j in pages){
    start <- as.character(j)
    search_url <- paste0(BASE_URL,jobTitle,'&l=',
                         location,'&filter=0&start=',start)
    print(search_url)
    # send get request, if it can't resolve to host, break loop
    resp <- tryCatch({expr=GET(search_url)},
                     error=function(search_url){return(NA)})
    print(resp[2])
    if (is.na(resp[2])) {print('Indeed does not like you anymore - change your IP!')}
    if (is.na(resp[2])) break
    resp_body <- read_html(resp) #only html
    

    # get job urls (sponsored)
    link_nodes <- html_nodes(resp_body, xpath =
                               paste0('//*[contains(concat( " ", @class, " " ),',
                                      ' concat( " ", "jobtitle", " " ))',
                                      ' and contains(concat( " ", @class, " " ),',
                                      ' concat( " ", "turnstileLink", " " ))]'))
    link_url_sponsored <- html_attr(link_nodes, name = "href")
    if (length(link_url_sponsored)>0)
    {
      link_url_sponsored <- paste0('https://www.indeed.com',
                                 link_url_sponsored)
    }

    link_nodes <- html_nodes(resp_body, xpath = 
                               paste0('//*[contains(concat( " ", @class, " " ),',
                                      ' concat( " ", "result", " " ))]//*[contains(concat( " ", @class, " " ),',
                                      ' concat( " ", "jobtitle", " " ))]//*[contains(concat( " ", @class, " " ),',
                                      ' concat( " ", "turnstileLink", " " ))]'))
    link_url <- html_attr(link_nodes, name = "href")
    link_url <- paste0('https://www.indeed.com',link_url) 
    #print(link_url)
    link_url <- append(link_url_sponsored,link_url)
    subQuery <- sapply(link_url, function(l){
      # extract job summary + title
      resp1 <- tryCatch(expr=GET(l), error=function(l){return(NA)})
      if (is.na(resp1[2])){
        return(cbind(0,0))
      }
      else {
      resp_body1 <- read_html(resp1)
      
      descNodes <- html_nodes(resp_body1,xpath = 
                                '//*[(@id = "job_summary")]')
      titleNodes <- html_nodes(resp_body1, xpath = 
                                 '//*[contains(concat( " ", @class, " " ), concat( " ", "jobtitle", " " ))]')
      titles_sub <- html_text(titleNodes)
      desc_sub <- html_text(descNodes)
      id_sub <- substr(l,nchar(l)-44,nchar(l)-6)
      return(cbind(titles_sub,desc_sub,id_sub))
      }
    })
    # it can happen that xpaths are empty and there we dont get proper output of the function -> just skip these 10 observations

    if(!is.null(dim(subQuery))){
    titles <- append(titles,subQuery[1,])
    desc <- append(desc,subQuery[2,])
    id <- append(id,subQuery[3,] )
    }
    else {
      print('Watch out, some data cannot be scrapped')
    }
    # break condition (if already on the max page then stop scrapping)
    breakNode <- html_nodes(resp_body, xpath =
                              '//*[contains(concat( " ", @class, " " ), concat( " ", "np", " " ))]')
    breakText <- html_text(breakNode)
    if ((grepl('Previous', breakText[1])==TRUE & length(breakText)==1)|(length(breakText)==0&j==0))
      {print("Maximum pages reached")}
    if ((grepl('Previous', breakText[1])==TRUE & length(breakText)==1)|(length(breakText)==0&j==0))
      break
    # try to randomize frecuency of get requests
    Sys.sleep(abs(rnorm(1,mean=2,sd=1)))
  }
  # format output
  searchTerm <- rep(gsub("+", "", jobTitle),length(titles))
  print(length(titles))
  print(length(searchTerm))
  print(length(desc))
  titles <- unname(titles)
  desc <- unname(desc)
  df_output <- data.frame(id,titles,searchTerm,desc)
  names(df_output) <- c('id','title','searchTerm','description')
  #return(title)
  return(df_output)
}



# extract jobs ----------------------------------------------------------
## add keyword + unique identifier ()
searchedJobs <- c('Business Developement','Sales','Finance', 'Consulting',
                  'Data Analyst','Machine Learning', 'Statistics','Data Scientist',
                  'Marketing', 'Artist', 'Market Research', 'Analytics', 'Human Ressources',
                  'Teacher', 'Psychology', 'Designer','Project management', 'Lawyer', 
                  'Supply chain','Engineer', 'Web Developer', 'Research', 
                  'Research and Developement ','Medicin', 'Doctor', 'Nurse','Service',
                  'Customer Experience','Business intelligence', 'Accounting',
                  'Wealth Management', 'Portfolio Manager', 'Pharmaceuticals',
                  'Biotech', 'Robotics', 'Hotel', 'Food', 'Electronic', 'Python', 'Javascript')


dataQuery <- lapply(searchedJobs, function(x){
 scrapper(x,'USA', 2000) 
})

dataQuery <-(rbindlist(dataQuery))
dataQuery <- unique(dataQuery)
x <-toJSON(dataQuery, pretty = TRUE)
rm(dataQuery)

write(x, 'trainingDataScrapped.json')

