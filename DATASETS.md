# Bases de dados

## Opção recomendada: usar o dataset já processado do projeto

Para comparar resultados com uma tabela de experimentos existente, o split
treino/val precisa ser **o mesmo**. Se o dataset já foi preparado em formato
YOLO em outro ambiente (pastas `images/{train,val}` + `labels/{train,val}` +
`data.yaml`), copie essa pasta e aponte `--data` para o `data.yaml`. Nenhum
download é necessário nesse caso.

## Download das bases originais

### Oxford Town Centre

A página original de Oxford foi retirada do ar; o dataset está disponível via:

- **Academic Torrents** (TownCentreXVID.mp4 86 MB + TownCentre-groundtruth.top
  3,7 MB + calibração):
  https://academictorrents.com/details/35e83806d9362a57be736f370c821960eb2f2a01
  (baixe o .torrent e abra num cliente BitTorrent, ex.: Transmission)
- **Kaggle** (espelhos; requer conta — busque "Oxford Town Centre")

Depois converta para o formato YOLO com o script deste repositório:

```bash
python prepare_towncentre.py --video TownCentreXVID.mp4 \
    --gt TownCentre-groundtruth.top --out oxford --train-frac 0.7
```

O script usa a caixa de corpo (bodyValid) do ground truth oficial e faz
**split temporal** (primeiros 70% dos frames → train, últimos 30% → val) para
evitar vazamento entre frames consecutivos. Use `--stride N` para amostrar
1 frame a cada N. **Atenção:** esse split não é necessariamente o mesmo de
uma tabela já preenchida — para comparação direta, use o dataset processado
original (opção acima).

### CAVIAR

Download direto (clips .mpg + ground truth XML por sequência):
https://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/

São dezenas de sequências curtas (corredor de shopping e saguão do INRIA). A
conversão XML→YOLO depende de quais sequências e classes o projeto usa, então
não há conversor genérico aqui — novamente, prefira o dataset processado do
projeto. Se precisar converter do zero, o ground truth de cada clip está no
link "Ground Truth" da própria página.

## Estrutura esperada pelo pipeline

```
oxford/
├── data.yaml          # path/train/val/names
├── images/
│   ├── train/*.jpg
│   └── val/*.jpg
└── labels/
    ├── train/*.txt    # YOLO: classe xc yc w h (normalizados)
    └── val/*.txt
```
