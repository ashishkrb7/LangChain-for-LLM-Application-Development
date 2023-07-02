# LangChain: Q&A over Documents

An example might be a tool that would allow you to query a product catalog for items of interest.


```python
#pip install --upgrade langchain
```


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.embeddings import OpenAIEmbeddings
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
```


```python
# Set OpenAI API key
os.environ["OPENAI_API_TYPE"] = os.getenv("api_type")
os.environ["OPENAI_API_BASE"] = os.getenv("api_base")
os.environ["OPENAI_API_VERSION"] = os.getenv("api_version")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```


```python
llm = AzureChatOpenAI(deployment_name="chatgpt-gpt35-turbo",model_name="gpt-35-turbo",temperature=0.0)
```


```python
from langchain.indexes import VectorstoreIndexCreator
```


```python
# pip install docarray
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding = OpenAIEmbeddings(model = "text-embedding-ada-002",chunk_size=1)
).from_loaders([loader])
```


```python
query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."
```


```python
response = index.query(query, llm)
```


```python
display(Markdown(response))
```


| Name | Description | Sun Protection |
| --- | --- | --- |
| Refresh Swimwear, V-Neck Tankini Contrasts | Watersport-ready tankini top designed to move with you and stay comfortable. | UPF 50+ rated – the highest rated sun protection possible. |
| Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece | Toddler's two-piece swimsuit with bright colors, ruffles, and exclusive whimsical prints. | UPF 50+ rated – the highest rated sun protection possible, blocking 98% of the sun's harmful rays. |

There is only two shirts with sun protection:
- Refresh Swimwear, V-Neck Tankini Contrasts: A watersport-ready tankini top designed to move with you and stay comfortable. It has UPF 50+ rated sun protection.
- Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece: A toddler's two-piece swimsuit with bright colors, ruffles, and exclusive whimsical prints. It has UPF 50+ rated sun protection, blocking 98% of the sun's harmful rays.



```python
loader = CSVLoader(file_path=file, encoding='utf-8')
```


```python
docs = loader.load()
```


```python
docs[0]
```




    Document(lc_kwargs={'page_content': ": 0\nname: Women's Campside Oxfords\ndescription: This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries.", 'metadata': {'source': 'OutdoorClothingCatalog_1000.csv', 'row': 0}}, page_content=": 0\nname: Women's Campside Oxfords\ndescription: This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries.", metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 0})




```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",chunk_size=1)
```


```python
embed = embeddings.embed_query("Hi my name is Harrison")
```


```python
print(len(embed))
```

    1536
    


```python
print(embed[:5])
```

    [-0.02186359278857708, 0.006734037306159735, -0.01820078119635582, -0.03919587284326553, -0.014047075994312763]
    


```python
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)
```


```python
query = "Please suggest a shirt with sunblocking"
```


```python
docs = db.similarity_search(query)
```


```python
len(docs)
```




    4




```python
docs[0]
```




    Document(lc_kwargs={'page_content': ": 5\nname: Smooth Comfort Check Shirt, Slightly Fitted\ndescription: Our men's slightly fitted check shirt is the perfect choice for your wardrobe! Customers love how it fits right out of the dryer. Size & Fit: Slightly Fitted, Relaxed through the chest and sleeve with a slightly slimmer waist. Fabric & Care: 100% cotton poplin, with wrinkle-free performance that won't wash out. Our innovative TrueCool® fabric wicks moisture away from your skin and helps it dry quickly. Additional Features: Traditional styling with a button-down collar and a single patch pocket. Imported.", 'metadata': {'source': 'OutdoorClothingCatalog_1000.csv', 'row': 5}}, page_content=": 5\nname: Smooth Comfort Check Shirt, Slightly Fitted\ndescription: Our men's slightly fitted check shirt is the perfect choice for your wardrobe! Customers love how it fits right out of the dryer. Size & Fit: Slightly Fitted, Relaxed through the chest and sleeve with a slightly slimmer waist. Fabric & Care: 100% cotton poplin, with wrinkle-free performance that won't wash out. Our innovative TrueCool® fabric wicks moisture away from your skin and helps it dry quickly. Additional Features: Traditional styling with a button-down collar and a single patch pocket. Imported.", metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 5})




```python
retriever = db.as_retriever()
```


```python
qdocs = "".join([docs[i].page_content for i in range(len(docs))])

```


```python
response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

```


```python
display(Markdown(response))
```


| Shirt Name | Sun Protection | Summary |
| --- | --- | --- |
| Refresh Swimwear, V-Neck Tankini Contrasts | UPF 50+ | This swimtop is made with recycled nylon and Lycra spandex for stretch and breathability. It has a flattering V-neck silhouette and racerback straps. Provides the highest rated sun protection possible. |
| Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece | UPF 50+ | This toddler's two-piece swimsuit has bright colors, ruffles, and exclusive prints. The four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. Provides the highest rated sun protection possible. |
| Smooth Comfort Check Shirt, Slightly Fitted | N/A | This men's check shirt is made with 100% cotton poplin and wrinkle-free performance. It has a slightly fitted, relaxed fit and traditional styling with a button-down collar and patch pocket. |
| EcoFlex 3L Storm Pants | N/A | These waterproof pants are made with TEK O2 technology for enhanced breathability. They have a slightly fitted fit and are ideal for a variety of outdoor activities year-round. Features include weather-blocking gaiters, side zips, and multiple pockets. |
 
Summary: The Refresh Swimwear and Infant and Toddler Girls' Coastal Chill Swimsuit both provide UPF 50+ sun protection, making them ideal for outdoor water activities. The Smooth Comfort Check Shirt and EcoFlex 3L Storm Pants do not have sun protection, but offer other features such as wrinkle-free performance and waterproof protection.



```python
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
```


```python
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."
```


```python
response = qa_stuff.run(query)
```

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    


```python
display(Markdown(response))
```


| Name | Description | Sun Protection |
| --- | --- | --- |
| Refresh Swimwear, V-Neck Tankini Contrasts | Watersport-ready tankini top designed to move with you and stay comfortable. Made with premium Italian-blend fabric that is breathable, quick-drying, and abrasion-resistant. UPF 50+ rated. | UPF 50+ rated |
| Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece | Two-piece swimsuit for toddlers with bright colors, ruffles, and exclusive whimsical prints. Made with four-way-stretch and chlorine-resistant fabric that keeps its shape and resists snags. UPF 50+ rated. | UPF 50+ rated |
 
Both the Refresh Swimwear, V-Neck Tankini Contrasts and Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece provide UPF 50+ rated sun protection. The Refresh Swimwear is a tankini top designed for watersports, made with premium Italian-blend fabric that is breathable, quick-drying, and abrasion-resistant. The Infant and Toddler Girls' Coastal Chill Swimsuit is a two-piece swimsuit for toddlers with bright colors, ruffles, and exclusive whimsical prints, made with four-way-stretch and chlorine-resistant fabric that keeps its shape and resists snags.

