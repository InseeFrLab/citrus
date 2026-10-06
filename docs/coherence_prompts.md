# Cohérence des prompts avec les règles métier

Comparaison de [`regles_metier.md`](regles_metier.md), le résumé de la spécification
« Intégration sources », avec `prompts/metier.md` (`metier-v3`) et `prompts/juridique.md`
(`juridique-v4`). Les observations chiffrées viennent du batch `2026-10-05_15-54-37`
(1 109 annonces, approche juridique).

Gravité : **haute** = fausse les prédictions ou l'évaluation d'un type entier ;
**moyenne** = touche une partie des cas ; **faible** = précision de formulation.

## 1. Prompt métier

### Ce qui est conforme

- L'ordre de l'arbre de décision : LG → TP → fusion/scission/apport à 2 UL → plus de 2 UL → VE
  → non retenu.
- La définition du SIREN potentiel.
- Les conditions TP et la façon de trouver le cédant et le bénéficiaire.
- Le code des cas à plus de 2 UL : AB si « absorption », sinon FU ; ST pour une scission.
- Le SIREN principal, une opération par SIREN secondaire, et la règle du « bon nombre de
  montants » sans repli.
- La cascade des montants à 2 UL, avec le repli sur le dernier montant.
- La cascade de dates de VE et ses montants.
- La cascade de dates de LG, et la règle « premier exploitant ».

### Écarts

| # | gravité | sujet | règle métier | prompt métier |
|---|---------|-------|--------------|---------------|
| M1 | haute | fusion ou scission à 2 UL | code provisoire AB/FZ ou SP/SZ, puis post-traitements sur l'ensemble des annonces : FZ → AB ou FU, SZ → SP ou ST, puis **FU ou ST isolé → AP** | « fusion » → toujours AB, « scission » → toujours SP |
| M2 | haute | dates des TP, fusions, scissions et apports dans les **annotations** | voir l'encadré ci-dessous | le prompt suit la spécification |
| M3 | moyenne | LG en RCS-A | le mot-clé est cherché dans `origineFonds` ou `categorieImmatriculation`, et la LG n'est retenue **que pour une immatriculation** | « n'exiger ce mot-clé que pour une immatriculation » : on comprend que, hors immatriculation, la LG est retenue sans mot-clé, ce qui inverse la règle |
| M4 | moyenne | précédent exploitant ou propriétaire | il doit avoir un **numéro d'immatriculation** ; un « RCS non inscrit » est exclu (cas de test OpA1) | « précédent exploitant non nul » suffit |
| M5 | moyenne | date d'effet des fusions, scissions et apports | la regex `effet (comptable|…`, tronquée dans le wiki, couvre au moins « **effet rétroactif au** » d'après le cas de test A3 | seulement « effet comptable » |
| M6 | faible | bénéficiaire d'une fusion à 2 UL | 7 ancres, dont « précédant est société bénéficiaire » et « précédant société absorbante » | 3 ancres seulement |
| M7 | faible | TP ou LG presque complet (second SIREN absent, fin de location-gérance) | rien n'est créé | le prompt invite à répondre `UNKNOWN` (retenu), alors que la règle métier ne retient pas l'annonce |
| M8 | faible | regex RCS-A pour plus de 2 UL | `…(fusion\|scission)\|projet[^.]{0,15}apport` | « même mot-clé », sans préciser que l'apport partiel n'a pas de code au-delà de 2 UL |

**M1 en détail.** Une seule annonce ne suffit pas pour reproduire ces post-traitements : ils
comparent plusieurs annonces de la même année. Le prompt fige donc AB et SP, alors que la
base contient des FU, des ST et surtout des **AP issus de fusions isolées**. Dans les cas de
test du document, deux absorptions explicites à 2 sociétés (A2 et A3) finissent en AP. Le
prompt ne peut pas suivre cette règle. Le mieux serait de l'appliquer dans le code, sur le
batch entier après l'extraction, comme le faisait l'ancien Citrus.

> **M2 — dates des annotations.** Selon la spécification, le web-service remplit une date de
> réalisation vide avec la date d'effet. Les annotations ne suivent pas toujours cette règle :
>
> | type annoté | ce que dit la spécification | ce qu'on observe |
> |---|---|---|
> | VE | réalisation = effet | réalisation = effet dans **300 cas sur 300** : conforme |
> | TP | effet = première date du descriptif, réalisation = dernière | les **deux dates sont vides dans 300 cas sur 300** |
> | AB, FU, SP, AP | réalisation = effet (imputée) | effet souvent vide (46/90 AB, 34/43 FU, 26/28 SP) et réalisation presque toujours remplie **et différente** de l'effet |
>
> Pour les TP, toute date prédite est donc comptée fausse. Pour les fusions, les deux colonnes
> semblent remplies selon une autre convention que la spécification, peut-être inversées.
> **Il faut le vérifier avec l'équipe qui produit `operations_verifiees.parquet` avant de
> conclure quoi que ce soit sur l'exactitude des dates.**

### Ce qui n'explique pas les FU annotées

On avait constaté que 30 des 43 FU annotées ne citent que 2 SIREN, alors que le prompt métier
réserve FU aux cas à plus de 2 UL. M1 donne un mécanisme possible : un FZ à 2 UL devient FU
quand plusieurs annonces partagent le même bénéficiaire. Mais la balise précédent propriétaire,
qui sert de critère entre AB et FZ, ne départage pas les annotations : il n'y en a pas pour
37 des 43 FU, ni pour 76 des 90 AB. Le classement FU des annotations ne se déduit donc pas
entièrement de la spécification. Une partie vient sans doute de la vérification manuelle
(fichier « opérations vérifiées ») ou des pratiques propres à chaque greffe.

## 2. Prompt juridique

Le prompt juridique décrit la nature des opérations, pas l'algorithme du Bodacc. On le compare
donc aux **définitions** du document (tableau de la section 1 de `regles_metier.md`), puis on
signale les endroits où l'algorithme métier contredit ces mêmes définitions.

### Définitions cohérentes

- **AB** : une absorbante qui existait déjà. Le document est conforme. Ses cas de test A2 et A3
  (filiale détenue à 100 %, aucun échange de titres) sont bien des absorptions et non des TP.
  Cela confirme la correction `juridique-v4`.
- **FU** : bénéficiaire créée, sociétés d'origine qui disparaissent. Conforme.
- **TP** : dissolution sans liquidation, parts réunies en une seule main, associé unique
  personne morale. **Conforme mot pour mot à `juridique-v4`.**
- **ST** : cédante qui disparaît, bénéficiaires existantes ou non. Conforme.
- **VE** : contrepartie purement financière. Conforme.
- **LG** : exploitation confiée puis récupérée. Conforme. Le document compte deux
  restructurations (début et fin), mais le Bodacc ne traite que le début.

### Incohérences

| # | gravité | sujet | document | prompt juridique |
|---|---------|-------|----------|------------------|
| J1 | moyenne | SP ou AP | **définitions** : SP = bénéficiaire(s) **créée(s)** pour l'occasion, AP = bénéficiaire qui **existait déjà** ; **algorithme** : le mot-clé (« scission » ou « apport partiel ») | le vocabulaire de l'acte décide (« projet de scission » → SP, « apport partiel » → AP) et « une bénéficiaire préexistante peut parfaitement figurer dans une scission partielle », ce qui contredit la définition de SP |
| J2 | moyenne | apport partiel « soumis au régime juridique des scissions » | AP (cas de test A4 : le mot-clé « apport partiel » l'emporte) | SP est défini comme un « apport d'une branche placé sous le régime juridique des scissions » : le cas A4 serait classé **SP** |

J1 et J2 posent la même question : faut-il distinguer SP et AP par la nouveauté de la
bénéficiaire (définition), par le mot-clé (algorithme), ou par le régime juridique (prompt) ?
Le document lui-même n'est pas cohérent sur ce point. Il faut trancher avec le métier.

### Écarts de convention, à ne pas compter comme des erreurs juridiques

Ces règles de l'algorithme contredisent les définitions du document lui-même. Le prompt
juridique, qui suit les définitions, s'en écartera donc forcément.

- **FU de l'algorithme ≠ FU de la définition.** Au-delà de 2 UL, une fusion sans le mot
  « absorption » est classée FU même si l'absorbante existait déjà, alors que la définition
  exige une bénéficiaire créée. C'est la cause de l'absence de FU prédites constatée dans le
  batch du 5 octobre (FU annotées prédites AB).
- **FU ou ST isolés → AP** (décision MOA « faute d'information ») : une absorption ou une
  scission juridiquement claire devient un apport partiel.
- **Scission à plus de 2 UL → ST par défaut** : une scission partielle à plusieurs
  bénéficiaires est classée ST.

Pour évaluer l'approche juridique sans pénaliser ces conventions, on pourrait regrouper FU, AB
et AP dans le calcul du « type exact » ou ajouter une métrique « famille fusion ». Ce n'est
pas fait à ce jour.
