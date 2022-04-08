

## TALLER WiDS 2022 - El discurso de género en el Parlamento uruguayo ##

#Instalo librerías
#De CRAN: 
#install.packages(c("speech","quanteda","dplyr","tidymodels","vip",
#"quanteda.textplots","RColorBrewer"))

#De GitHub: 
#remotes::install_github("Nicolas-Schmidt/puy")


#Cargo librerías
library(speech)
library(puy)
library(quanteda)
library(dplyr)
library(tidymodels)
library(vip)



#1. Obtención de la información

#Para obtener las url de las sesiones parlamentarias
# S Camara de Senadores; D Camara de Representantes; A Asamblea General; C Comisión Permanente

urls = speech::speech_url(chamber  = "D",
            from        = "01-04-2015",
            to          = "01-05-2015")

intervenciones = speech::speech_build(urls)


##ATENCIÓN!## 
#Para descargar una cantidad grande de sesiones, se sugiere usar la función map de purrr 
#para que no se corte la descarga en caso de encontrar pdf con problemas de transcripción 
# discursos=map(url,possibly(speech_build,otherwise = NULL))
# discursos_df = discursos %>% 
# map_df(as_tibble)


# uso la función puy::add_party para agregar la etiqueta partidaria

intervenciones = puy::add_party(intervenciones)


# Cargo la base de la legislatura 2015-2020 (legislatura 48) correspondiente a la Cámara de Diputados/as 
# ya descargada

load("Datos/intervenciones_2015_2020.RData")

# Cargo la base de la Comisión de Género que me va a servir para construir el modelo de aprendizaje. 

load("Datos/com_genero.RData")



#2. Limpieza del texto y matriz de términos
  
## Empiezo con la limpieza del corpus de datos de la Comisión de Género 
#y creación de matriz de términos


##En primer lugar, quito las sentencias con menos de 13 palabras
speech <- quanteda::corpus(com_genero,text_field = "speech")%>%
  quanteda::corpus_trim(what = "sentences",min_ntoken = 13) 
com_genero=cbind(docvars(speech),speech)

##Armo la matriz de términos, haciendo limpieza

dfm_com_genero<- quanteda::dfm(quanteda::tokens(com_genero$speech,
                                    remove_punct = TRUE,
                                    remove_numbers = TRUE),
                                    tolower=TRUE,
                                    verbose = FALSE) %>%
  quanteda::dfm_remove(pattern = c(quanteda::stopwords("spanish"), 
                                   tolower(com_genero$legislator),
                                   "señor","señora","presidente","presidenta"), 
                       min_nchar = 3)%>%
        quanteda::dfm_trim(min_termfreq = 60)


##Evalúo en nube de palabras
quanteda.textplots::textplot_wordcloud(quanteda::dfm_group(dfm_com_genero,groups = com_genero$genero), 
                                       min.count = 30, 
                                       max_words = 300,
                                       random.order = FALSE ,
                                       rot.per = .25,
                                       colors = RColorBrewer::brewer.pal(8,"Dark2"),
                                       comparison = TRUE)


# 3. Clasificación: aprendizaje automático y diccionario


## Armo un data frame a partir de la matriz de términos y le agrego la variable 
#genero con: Si genero - No genero

dfm_com_genero_df <- dfm_com_genero %>%
  convert(to = "data.frame") %>%
  mutate(genero=docvars(dfm_com_genero, "genero"))


set.seed(123)
genero_split <- initial_split(dfm_com_genero_df, strata = genero)
genero_train <- training(genero_split)
genero_test <- testing(genero_split)

genero_rf <-  rand_forest(trees = 350, mode = "classification") %>%
  set_engine("ranger",importance = "impurity") %>%
  fit(genero ~ ., data = genero_train[, !(colnames(genero_train) %in% c("doc_id"))])


#save(afam_rf,file="Bases/afam_rf.RData")


#ENTRENO

genero_train = genero_rf %>%
  predict(genero_train) %>%
  bind_cols(genero_train)

library(vip)

##ploteo importantes
genero_rf%>%
  vip(num_features = 20)


table(genero_train$genero,genero_train$.pred_class)

metrics(genero_train,truth = genero, estimate = .pred_class)


#TESTEO

genero_test = genero_rf %>%
  predict(genero_test) %>%
  bind_cols(genero_test)


##Uso el modelo para predecir en la base grande, me quedo con las variables comunes

## En primer lugar, creo la matriz de términos de la base de intervenciones,
# de la misma forma que lo hice con la comisión de género.  

speech <- quanteda::corpus(intervenciones_2015_2020,text_field = "speech")%>%
  quanteda::corpus_trim(what = "sentences",min_ntoken = 13) 
diputados=cbind(docvars(speech),speech)

##Armo la matriz de términos, haciendo limpieza

dfm_diputados<- quanteda::dfm(quanteda::tokens(diputados$speech,
                              remove_punct = TRUE,
                              remove_numbers = TRUE),
                               tolower=TRUE,
                               verbose = FALSE) %>%
  quanteda::dfm_remove(pattern = c(quanteda::stopwords("spanish"), 
                                   tolower(com_genero$legislator),
                                   "señor","señora","presidente","presidenta"), 
                       min_nchar = 3)%>%
  quanteda::dfm_trim(min_termfreq = 20) ## no soy tan restrictiva


dfm_diputados_df <- dfm_diputados %>%
  convert(to = "data.frame")

dfm_diputados_df[,setdiff(colnames(genero_train),colnames(dfm_diputados_df))]=0

dfm_diputados_df <- dfm_diputados_df %>%
  select(colnames(genero_train))%>%
 select(-.pred_class, -genero)


dfm_diputados_df = genero_rf %>%
  predict(dfm_diputados_df) %>%
  bind_cols(dfm_diputados_df) %>%
  select(.pred_class)%>%
  bind_cols(diputados)%>%
  filter(.pred_class=="Si genero")
  

##Vuelvo a realizar una matriz de términos, con las intervenciones pre-clasificadas


dfm_sigenero<- quanteda::dfm(quanteda::tokens(dfm_diputados_df$speech,
                             remove_punct = TRUE,
                             remove_numbers = TRUE),
                              tolower=TRUE,
                              verbose = FALSE) %>%
  quanteda::dfm_remove(pattern = c(quanteda::stopwords("spanish"), 
                                   tolower(dfm_diputados_df$legislator),
                                   "señor","señora","presidente","presidenta"), 
                       min_nchar = 3)



##Creo diccionario de "género" a partir de las 50 palabras más mencionadas 
#en el dfm de Comisión de género 

dicgenero <- dfm_com_genero %>%
quanteda::topfeatures(50,groups = genero)
dicgenero <- rownames(as.data.frame(dicgenero[[2]]))
dicgenero <-dictionary(list(genero=dicgenero))

##aplico diccionario y genero un ratio que es la cantidad de palabras en el
##diccionario entre la cantidad de palabras totales. Me quedo con las intervenciones
#que tienen más de 5% de palabras vinculadas al tema.  

resultado_dic <- data.frame(dfm_lookup(dfm_sigenero,dictionary=dicgenero))

 df_sigenero = dfm_diputados_df %>%
  bind_cols(resultado_dic) %>%
  mutate(ratio=genero/words)%>%
   filter(ratio>=0.05)


 
 
 
#4. Análisis de los datos 
 

 #Grafico por sexo 
 
   genero=df_sigenero %>%
     mutate(sexo=ifelse(sex==1,"Varón","Mujer")) %>%
     group_by(sexo) %>%
     summarise(ratio= n()/length(unique(legislator2)))%>%
     ggplot(aes(x = reorder(sexo, -ratio) , y = ratio,group = 1)) + 
     geom_bar(size=1, stat="identity",fill = c("#e76363","#f09e9e"),width = 0.6) +
     scale_y_continuous(limits = c(0, 12))+ 
     theme(axis.text.x = element_text(size=10), 
           axis.text.y = element_text(size=10),
           axis.title.x=element_blank(),axis.title.y=element_blank())
 

   #Grafico por partido 
   

   partido=df_sigenero %>%
     group_by(party_acron) %>%
     drop_na()%>%
     summarise(conteo=n())%>%
     ggplot(aes(x = reorder(party_acron, -conteo) , y = conteo,group = 1)) + 
     geom_bar(size=1, stat="identity",fill = c("#954342","#954342","#4a2524","#e76363","#bd5352","#6e3432"),width = 0.6) +
     #scale_y_continuous(limits = c(0, 12))+ 
     theme(axis.text.x = element_text(size=10), 
           axis.text.y = element_text(size=10),
           axis.title.x=element_blank(),axis.title.y=element_blank())
   

   #Grafico por legislador/a 
   
   
   legis=df_sigenero %>%
     count(legislator2)%>%
     drop_na()%>%
     top_n(n=10) %>%
     arrange(n)%>%
     ggplot(aes(x = reorder(legislator2, n) , y = n,group = 1)) + 
     geom_bar(size=1, stat="identity",fill = c("#954342"),width = 0.6) +
     scale_fill_gradient(low = "white", high = "red")+
     coord_flip()+
     theme(axis.text.x = element_text(size=10), 
           axis.text.y = element_text(size=10),
           axis.title.x=element_blank(),axis.title.y=element_blank())
   
