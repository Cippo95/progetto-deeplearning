#Commento in data 22/12/2021

Questo è il mio progetto di deep learning concluso in data 27/01/2020.
Ricarico questo progetto su GitHub a fini di backup.
Non inserisco però il dataset che mi era stato fornito dal professore.

##PROGETTO DI DEEP LEARNING - FILIPPO LANDI matr. 121120 - UNIFE

Nel seguente file parlo dei principali punti del progetto, più o meno in ordine cronologico di
implementazione:

###CONSIDERAZIONI INIZIALI

Ho installato Python 3.7 anche se la versione 3.8 è già disponibile ma tensorflow non lo era ancora (al tempo)
per tale versione.  
Ho installato diverse librerie, tenforflow appunto, in dettaglio per tensorflow-gpu 2.1.0 per la mia scheda video Nvidia;
opencv-python e tante altre... quindi se il codice non gira è bene guardare le varie richieste delle librerie.  
Tensorflow 2.1.0 sembra dare tanti warning, all'inizio mi ero preoccupato ma leggendo online è normale, ad esempio:
https://github.com/tensorflow/tensorflow/issues/35100
https://stackoverflow.com/questions/59317919/warningtensorflowsample-weight-modes-were-coerced-from-to
Per lanciare il programma serve indicare l'argomento '-d nomecartellaallenamento' che nel nostro caso è 'Digit'.
Poi ci sono altri argomenti come '-a numeromaggiore di 0' per indicare se si vuole aggiungere l'augmentation (default -1),
'-n numerointeromaggioredi 0' per il numero di turni (default 1), '-p nomeplot' per salvare un grafico delle loss e accuracy su validation e training,
che viene comunque salvato ma con il generico nome 'plot'.
Dentro AUG_D ci sono degli esempi di dati aumentati per vedere il risultato dovuto alla trasformazione.

###IMPLEMENTAZIONE DELLA RETE

Essenzialmente fine tuning di una rete data in un tutorial di image recognition della Google, 
eseguiva una classificazione binaria per classificare cani e gatti.
Ho quindi messo uno strato di uscita con 10 uscite, funzione softmax invece che semplice sigmoide
e come loss la categorical_crossentropy e ridefinito la dimensione delle immagini per il mio dataset.
In pratica sono passato da classificazione binaria a classificazione multiclasse con le solite funzioni studiate.
Inizialmente avevo tenuto la rete completamente connessa con 512 neuroni interni ma anche a 256 funziona bene,
mentre a 128 ho notato allenando con SGD della 'varianza' nell'allenamento.
Come ottimizzatori ho provato sia la adam con parametri di default che la SGD con momento.
La adam converge molto più in fretta ma la SGD sembra far raggiungere alla rete una precisione maggiore dopo 100 'turni' di allenamento,
dico turni perché ho messo che le epoche sono fatte per ogni fold mentre i turni sono un ciclo esterno che fa ripetere il procedimento.  
Non ho giocato molto con gli ottimizzatori perché mi pareva che i risultati fossero già buoni ma per maggiori info qui ci sono varie opzioni:
https://keras.io/optimizers/
Come detto mi sono fermato qui perché la rete in test funzionava già bene 96-98% accuracy a seconda dell'allenamento, parametri leggermente 
differenti etc.  
Anche per questo non ho preso in considerazione early stopping, semplicemente salvo dopo 100 turni (che si possono indicare a riga come
argomento con '-n numeroturni').
Per maggiori informazioni sulla rete iniziale della Google:
https://www.tensorflow.org/tutorials/images/classification

###SETUP DATI E DATA AUGMENTATION

Inizialmente usavo la libreria di keras 'ImageDataGenerator' poiché davvero semplice per leggere le immagini 
e caricarle nel modello con label ed eseguire data augmentation.
https://keras.io/preprocessing/image/

Successivamente per le problematiche riscontrate con il mio desiderio di implementare la kfold sono passato
ad una implementazione più classica che appoggia su scikit learn e opencv.
Le problematiche erano che con la ImageDataGenerator si può risevare una validation dal training, ma questa risulta
statica nel senso che viene scelta e rimane la stessa per tutto l'allenamento oltre il fatto che vengono purtroppo
fatte anche su di essa le trasformazioni dovute al data augmentation cosa che invece non succede caricando i dati
tramite scikit learn e usando la kfold validation in quanto riesco a variare appunto il gruppo di validatione e 'targettare'
correttamente i dati di training.  
Basta cercare online "ImageDataGenerator cross-validation" per vedere molta gente con lo stesso problema.
Un link che dove la risposta principale è "bisogna fare la cross validation manualmente": 
https://stackoverflow.com/questions/55963087/in-keras-imagedatagenerator-is-validation-split-parameter-a-kind-of-k-fold

ImageDataGenerator rimane comunque per eseguire la data augmentation sui dati di training con varie trasformazioni, 
tramite quelle che chiamo 'opzioni di debug' in inglese quindi 'debug option' che ho commentato di default si può fare in modo
di salvare i dati trasformati così da vederli e giudicare se possono andare bene come dati su cui fare allenare 
la rete poiché certe volte reputo che le trasformazioni siano un po' troppo distruttive, per esempio tra le varie
opzioni non ho messo il flip orizzontale perché non mi pareva sensato.  
Nel primo link di questa sezione ci sono scritte le varie trasformazioni se si vogliono maggiori info.

###DATA AUGMENTATION PIÙ IN SPECIFICO
  
Ho voluto eseguire un aumento "on the fly" cioè una modifica dei dati mentre venivano dati alla rete.
Quindi vengono presi i dati di training e trasformati casualmente con i parametri dati e dati in pasto alla rete.
Articolo importantissimo che ho usato per fare questo è al seguente link:
https://www.pyimagesearch.com/2019/07/08/keras-imagedatagenerator-and-data-augmentation/
Da questo articolo inoltre deriva buona parte del progetto: l'implementazione di argomenti, come leggere i dati e dividere
il training e il test set, ottimizzatore SGD (anche se adam funzionava benissimo), plot e valutazioni.

###IMPLEMENTAZIONE KFOLD

Ho letto un po' di questo 'tutorial' e ho implementato la kfold vanilla:
https://scikit-learn.org/stable/modules/cross_validation.html
Ho lasciato i parametri di default (5 fold) tranne il parametro shuffle attivato poiché migliora il modo in 
cui funziona la validazione, se no all'inizio la validazione opererebbe su dati completamente non visti: si può togliere
lo shuffle e si vedrà che all'inizio l'accuracy partirà da 0 (non proprio ma un valore molto piccolo) per questo.

###MODIFICA MANUALE DEL DATASET

Ho preso come modello il dataset MNIST dove c'è un rapporto 6:1 tra i dati di training e test.
Ho notato che gli esempi originali di test del numero 7 erano contenuti nel training, non andava bene,
in quanto i dati di test devono non essere visti durante il training.
Ho ripartizionato i dati con occhio alle varie posizioni dei numeri (riguardo a numero precedente e successivo)
facendo in modo che nel training ci fossero 50-60 esempi per categoria e nel test almeno sulla decina di esempi.
Il numero 7 mi è rimasto con solo 8 elementi di test ma non avendo molte foto ho preferito lasciare più dati nel 
training e scegliere delle immagini che mi sembravano le più appropriate per il test.
Per info sui dati MNIST:
http://yann.lecun.com/exdb/mnist/

###SALVATAGGIO RETE

Implementazione molto molto semplice che salva i parametri allenabili della rete a fine allenamento, 
si può allenare la rete o caricare i pesi e vedere le statistiche della rete sui dati di test.  
Si potrebbero implementare salvataggi più pervasivi ma questo mi sembrava abbastanza, per maggiori info:
https://www.tensorflow.org/tutorials/keras/save_and_load











