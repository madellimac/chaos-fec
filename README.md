# Génération d'une image chaotique  
## Présentation des différents fichiers  
Cette branche contient les fichiers suivants :  
1. __Sequence.ipynb__ :    
    * Programme principal fonctionnel. Il est structuré comme suit :
        * La __séquence 1__ génère le vecteur à décoder. Il est possible de sauvegarder ce vecteur en mémoire pour le charger ensuite.  
        * La __séquence 2__ utilise les modules *conductor* et *add_impulses* décrit en pyaf (C++) et génère l'image (module *display*).  
2. __factory.py__ :  
    * Ce script permet de définir les informations liées à la __séquence 1__ :   
        * Type d'encodage utilisé,  
        * Déclaration de l'encodeure,
        * Déclaration du décodeur,  
        * Déclaration des constantes associée (K, N, Eb).
3. __Pipeline.ipynb__ : 
    * C'est le même programme que __Sequence.ipynb__ mais il utilise le pipeline pour la __séquence 2__. (*__NON FONCTIONNEL__*)  
4. __GUI.py__ :  
    * L'interface graphique déclare les sequences dans une methode affichant l'image.  
---------------------------------------------------------------------------------------
## Le module *conductor* est désormais décrit en C++ avec pyaf.  
Ce module fournit au module *add_impulses* le __llr__ (noisy_vec) ainsi que les informations relatives à ce vecteur à modifier. De plus, *conductor* donne au module *display* les positions __x__ et __y__ dans l'image.  
Ce module peut être récupéré dans le fork pyaf de greg-lee-surf dans le dossier [Conductor](https://github.com/greg-lee-surf/pyaf/tree/master/src/cpp/Module/Conductor).

## Le module *add_impulses* est désormais décrit en C++ avec pyaf.  
Ce modoule récupère toutes les informations fournit par *conductor* et modifie le vecteur d'entrée. 
Ce module peut être récupéré dans le fork pyaf de greg-lee-surf dans le dossier [Add](https://github.com/greg-lee-surf/pyaf/tree/master/src/cpp/Module/Add).  

## Le module *display* est désormais décrit en C++ avec pyaf.  
Ce module construit l'image *chaotique*. De plus, c'est ici qu'est envoyé le signal permettant d'arrêter la séquence 2.  
Ce module peut être récupéré dans le fork pyaf de greg-lee-surf dans le dossier [Display](https://github.com/greg-lee-surf/pyaf/tree/master/src/cpp/Module/Display).  

---------------------------------------------------------------------------------------

## L'interface graphique  
L'interface graphique tient dans le fichier *GUI.py*. Pour l'instant, cette GUI contient une unique classe. Elle redéfinit les modules, les sockets ainsi que les séquences dans des méthodes. Ces méthodes sont appelées en fonction des actions de l'utilisateur.
