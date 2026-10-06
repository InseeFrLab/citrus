# Règles métier de classification des restructurations

Résumé de la spécification « Intégration sources » du wiki Citrus
(dernière modification : 11 août 2022), limité à ce qui sert à classer
et à décrire une opération de restructuration. Les parties sur les sources Esane, Oreste, TSE2G,
Conjoncture et Harmonica, et les jeux de tests, ne sont reprises que lorsqu'elles éclairent
une règle.

## 1. Les huit types

| code | type | définition du document |
|------|------|------------------------|
| AP | apport partiel d'actif | le ou les cédants existent encore après la restructuration, le ou les bénéficiaires existaient avant ; souvent l'apport d'une branche |
| SP | scission partielle | la cédante cède une partie de ses actifs à une ou des entreprises **créées à cette occasion** |
| ST | scission totale | la cédante disparaît ; ses actifs vont à plusieurs entreprises, existantes ou non |
| AB | absorption | la ou les cédantes sont absorbées par une entreprise qui **existait déjà** |
| FU | fusion | la bénéficiaire est **créée** à partir de la fusion de plusieurs entreprises qui disparaissent |
| TP | transmission universelle de patrimoine | dissolution d'une société dont toutes les parts sont réunies en une seule main, avec transmission de son patrimoine à l'associé unique (personne morale), sans liquidation |
| LG | location-gérance | une entreprise confie des actifs à une autre pendant un bail (souvent 3 ans) puis les récupère : deux restructurations, une au début et une à la fin |
| VE | vente | apport d'actifs dont la contrepartie est purement financière |

Règles générales :

- Un type inconnu est laissé **vide** (recodé `IM`, « imprécis », pour la livraison à Esane).
- Deux montages rares, la scission partielle-fusion et la scission-absorption, sont classés
  différemment selon les sources.
- **Une seule opération par couple d'unités légales et par année de campagne** (décision MOA).
- Une même restructuration donne souvent une annonce par unité légale, avec le même descriptif :
  les doublons sont filtrés ensuite par le web-service.

## 2. Notions communes

- **SIREN potentiel** : 9 chiffres, éventuellement séparés par des espaces ou des points,
  non précédés d'un chiffre, non suivis de « € », « eur », d'un chiffre, de « action »,
  d'un point ou d'une virgule, et qui valident la clé de Luhn.
- **SIREN objet** : celui de la personne objet de l'annonce
  (`personnes/personne/…/numeroIdentification` en RCS-A,
  `personnes/personne/numeroImmatriculation/numeroIdentificationRCS` en RCS-B).
  Les « non inscrits » sans SIREN sont exclus.
- Expressions régulières et mots-clés insensibles à la casse.
- Année de campagne : année du fichier du Bodacc (corrigée ensuite selon la date d'effet).

## 3. Arbre de décision Bodacc

Les cas sont testés dans cet ordre. La vente n'est retenue que si aucun cas précédent ne l'a été.

### 3.1 Location-gérance → LG

| | RCS-A | RCS-B |
|---|---|---|
| bénéficiaire | SIREN objet non vide | SIREN objet non vide |
| cédant | précédent **exploitant** (PM ou PP) **avec un numéro d'immatriculation** | idem, dans `modificationsGenerales` |
| mot-clé `location[ -]?g[eé]rance` | dans `origineFonds` ou `categorieImmatriculation`, **seulement pour une immatriculation** (`acte/immatriculation`) | dans `modificationsGenerales/descriptif` |

- Le rachat du fonds par l'exploitant (précédent **propriétaire**) relève de la vente.
- S'il y a plusieurs précédents exploitants, on prend le premier.
- Une fin de location-gérance (cessation avec un seul SIREN) ne donne rien.

### 3.2 Transmission universelle de patrimoine → TP (RCS-B seulement)

- SIREN objet non vide : c'est le **cédant**, la société dissoute.
- Le descriptif contient `transmission universelle d[ue] patrimoine` ou `transmiss.univers.patrimoine`.
- Le descriptif contient un **autre SIREN potentiel** : c'est le **bénéficiaire**, l'associé unique.
  Sans ce second SIREN, l'annonce ne donne rien.

### 3.3 Fusion, scission, apport partiel entre 2 unités légales → AB/FZ, SP/SZ, AP

Condition : deux SIREN potentiels au plus, le SIREN objet et un autre dans le descriptif.

- **RCS-A** : la concaténation de `origineFonds` et `acte/vente/descriptif` correspond à
  `projet[^.]{0,50}(fusion|scission|apports? partiels?)|projet[^.]{0,15}apport`.
- **RCS-B** : `modificationsGenerales/descriptif` correspond à
  `fusion|scission|apport partiel d'actif`.

Code provisoire selon le mot-clé reconnu :

- « fusion » → **AB** ou **FZ** (fusion indéterminée) ;
- « scission » → **SP** ou **SZ** (scission indéterminée) ;
- « apport partiel » → **AP**.

**Condition perdue dans le wiki.** Dans le source Markdown de la page, chaque ligne de ce
tableau s'arrête à `« projet[^\\.]{0,50}(fusion |` : le `|` de l'expression régulière ferme la
cellule et le reste de la condition est perdu. Le critère qui sépare AB de FZ, et SP de SZ, n'est
donc écrit nulle part. D'après le §3 du document (repris en section 4 ci-dessous), ce serait la
**balise précédent propriétaire égale au SIREN de l'annonce** : présente, elle donne AB ou SP ;
absente, FZ ou SZ. **C'est une reconstitution, à faire confirmer par le métier.**

**Bénéficiaire**, par ordre de priorité :

- **scission** :
  1. SIREN après « société bénéficiaire de la scission : » ;
  2. après « société bénéficiaire : » ;
  3. avant « est société bénéficiaire » ;
  4. (RCS-B) l'autre SIREN que l'objet ;
  5. sinon le premier SIREN.
- **apport partiel** :
  1. le SIREN objet s'il n'y a qu'un SIREN dans le descriptif ;
  2. après « société bénéficiaire de l'apport : » ;
  3. après « société bénéficiaire : » ;
  4. avant « est société bénéficiaire » ;
  5. sinon le deuxième SIREN (l'apporteuse est citée en premier).
- **fusion** :
  1. avant « est société absorbante » ;
  2. après « société absorbante : » ;
  3. avant « est société bénéficiaire » ;
  4. après « pour la société absorbante » ;
  5. avant « société absorbante » ;
  6. (RCS-B) le SIREN objet ;
  7. sinon le premier SIREN.

**Cédant** : le SIREN objet s'il n'est pas le bénéficiaire, sinon l'autre SIREN du descriptif.

### 3.4 Fusion ou scission avec plus de 2 unités légales → AB / FU / ST

Condition : au moins deux autres SIREN potentiels que le SIREN objet dans le descriptif.

- **RCS-A** : la regex est `projet[^.]{0,50}(fusion|scission)|projet[^.]{0,15}apport`.
- **RCS-B** : la regex est `fusion|scission`.

**SIREN principal**, par ordre de priorité :

- **scission** (c'est le cédant) :
  1. après « société scindée : » ;
  2. sinon le premier SIREN.
- **fusion** (c'est le bénéficiaire) :
  1. après « Société absorbante : » ;
  2. avant « est société bénéficiaire » ;
  3. (RCS-B) le SIREN objet ;
  4. sinon le premier SIREN.

On crée **une opération par SIREN secondaire**, avec ce code :

- « fusion » et le mot « **absorption** » dans le descriptif (souvent « fusion par voie
  d'absorption ») → **AB** ;
- « fusion » sans le mot « absorption » → **FU**, le cas par défaut ;
- « scission » → **ST** : en Bodacc, on ne sait pas distinguer scission totale et partielle
  quand il y a beaucoup d'unités légales.

### 3.5 Autres ventes → VE (RCS-A seulement, si aucun cas précédent)

- SIREN objet non vide : c'est l'**acheteur**, le bénéficiaire.
- Un précédent **propriétaire** (PM ou PP) avec un numéro d'immatriculation : c'est le cédant.
- Une balise `vente`.

## 4. Post-traitements globaux, sur l'ensemble des annonces

Ces règles comparent **plusieurs annonces entre elles**. Une lecture annonce par annonce ne
peut donc pas les reproduire.

1. **Absorption ou fusion, scission partielle ou totale** (cas à 2 unités légales).
   Une branche dont le précédent propriétaire est le SIREN de l'annonce signale une absorption
   ou une scission partielle. Une vraie fusion ou une scission totale implique plus de deux
   unités légales et n'a pas cette balise.
   - FZ ayant le même bénéficiaire qu'une opération AB → **AB** ;
   - SZ ayant le même cédant qu'une opération SP → **SP** ;
   - FZ restants → **FU** ; SZ restants → **ST**.
2. **Fusion ou scission totale isolée → AP** (décision MOA, comme dans l'ancien Citrus).
   - Une FU qui ne partage son bénéficiaire avec aucune autre FU de la même année devient **AP**.
   - Une ST qui ne partage son cédant avec aucune autre ST de la même année devient **AP**.
   - Raison : en RCS-B, il s'agit souvent d'une modification (nom, activité) liée à une fusion
     ancienne, dont on ne voit qu'une branche. Les cas de test A2 et A3 du document (deux
     absorptions à 2 sociétés) finissent ainsi en AP.
3. **Couple cédant = bénéficiaire supprimé**. Ce couple a servi à repérer l'absorption ou la
   scission partielle, mais il ne porte pas le montant.

Le document reconnaît que ces règles viennent de l'observation de cas unitaires et que les
pratiques des greffes sont hétérogènes.

## 5. Champs extraits

### Date d'effet comptable

| type | par ordre de priorité |
|------|-----------------------|
| LG | `dateCommencementActivite` → `dateEffet` → (RCS-B) date après « à compter du » → date du bulletin |
| TP | première date du descriptif → date du bulletin |
| AB / FZ / SP / SZ / AP, et > 2 UL | date après `effet (comptable|…` (la regex est tronquée dans le wiki ; d'après le cas de test A3, elle couvre au moins « effet rétroactif au ») → (RCS-B) `dateCommencementActivite` → (RCS-B) `dateEffet` → date après « à compter du » → date du bulletin ; pour plus de 2 UL, la dernière en cas d'ambiguïté |
| VE | date du journal d'annonces légales → `dateCommencementActivite` → `dateEffet` → seule date du descriptif de vente (si pas de `dateEffet`) → date du bulletin |

### Date de réalisation juridique

- **LG** : en RCS-A, la dernière date du descriptif ; en RCS-B, vide.
- **TP** : la dernière date du descriptif.
- **Autres types** : vide.
- Une date vide est **imputée par le web-service à partir de la date d'effet**. En base,
  la date de réalisation n'est donc jamais vide.

### Montant net

Le montant est en k€, sans décimale.

- **LG et TP** : vide.
- **Fusion, scission ou apport à 2 UL**, montant pris après la première expression trouvée
  dans cet ordre :
  1. « Actif net apporté : » ;
  2. « actif net apporté égal à » ;
  3. « La valeur nette des apports s'élèverait à » ;
  4. « La valeur nette positive des apports s'élèverait à : » ;
  5. « actif : » ;
  6. « actif de » ;
  7. sinon le dernier montant du descriptif, qui est souvent la prime de fusion.
- **Plus de 2 UL** : on applique les mêmes expressions (1 à 6), mais on ne retient que
  la première qui donne autant de montants que de SIREN secondaires. Si aucune ne convient,
  pas de montant : il n'y a pas de repli sur le dernier montant.
- **VE** : le montant dans `origineFonds`, sinon le dernier montant du descriptif de vente.
