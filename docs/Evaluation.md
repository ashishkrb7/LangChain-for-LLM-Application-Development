# LangChain: Evaluation

## Outline:

* Example generation
* Manual evaluation (and debuging)
* LLM-assisted evaluation


```python
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
```


```python
# Set OpenAI API key
os.environ["OPENAI_API_TYPE"] = os.getenv("api_type")
os.environ["OPENAI_API_BASE"] = os.getenv("api_base")
os.environ["OPENAI_API_VERSION"] = os.getenv("api_version")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

## Create our QandA application


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.embeddings import OpenAIEmbeddings
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file, encoding='utf-8')
data = loader.load()
```


```python
embeddings = OpenAIEmbeddings(model = "text-embedding-ada-002",chunk_size=1)
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding = embeddings
).from_loaders([loader])
```


```python
llm = AzureChatOpenAI(deployment_name="chatgpt-gpt35-turbo",model_name="gpt-35-turbo",temperature=0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```

### Coming up with test datapoints


```python
data[2]
```




    Document(lc_kwargs={'page_content': ": 2\nname: Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece\ndescription: She'll love the bright colors, ruffles and exclusive whimsical prints of this toddler's two-piece swimsuit! Our four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. Machine wash and line dry for best results. Imported.", 'metadata': {'source': 'OutdoorClothingCatalog_1000.csv', 'row': 2}}, page_content=": 2\nname: Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece\ndescription: She'll love the bright colors, ruffles and exclusive whimsical prints of this toddler's two-piece swimsuit! Our four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. Machine wash and line dry for best results. Imported.", metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 2})




```python
data[3]
```




    Document(lc_kwargs={'page_content': ": 3\nname: Refresh Swimwear, V-Neck Tankini Contrasts\ndescription: Whether you're going for a swim or heading out on an SUP, this watersport-ready tankini top is designed to move with you and stay comfortable. All while looking great in an eye-catching colorblock style. \n\nSize & Fit\nFitted: Sits close to the body.\n\nWhy We Love It\nNot only does this swimtop feel good to wear, its fabric is good for the earth too. In recycled nylon, with Lycra® spandex for the perfect amount of stretch. \n\nFabric & Care\nThe premium Italian-blend is breathable, quick drying and abrasion resistant. \nBody in 82% recycled nylon with 18% Lycra® spandex. \nLined in 90% recycled nylon with 10% Lycra® spandex. \nUPF 50+ rated – the highest rated sun protection possible. \nHandwash, line dry.\n\nAdditional Features\nLightweight racerback straps are easy to get on and off, and won't get in your way. \nFlattering V-neck silhouette. \nImported.\n\nSun Protection That Won't Wear Off\nOur high-performance fabric provides SPF", 'metadata': {'source': 'OutdoorClothingCatalog_1000.csv', 'row': 3}}, page_content=": 3\nname: Refresh Swimwear, V-Neck Tankini Contrasts\ndescription: Whether you're going for a swim or heading out on an SUP, this watersport-ready tankini top is designed to move with you and stay comfortable. All while looking great in an eye-catching colorblock style. \n\nSize & Fit\nFitted: Sits close to the body.\n\nWhy We Love It\nNot only does this swimtop feel good to wear, its fabric is good for the earth too. In recycled nylon, with Lycra® spandex for the perfect amount of stretch. \n\nFabric & Care\nThe premium Italian-blend is breathable, quick drying and abrasion resistant. \nBody in 82% recycled nylon with 18% Lycra® spandex. \nLined in 90% recycled nylon with 10% Lycra® spandex. \nUPF 50+ rated – the highest rated sun protection possible. \nHandwash, line dry.\n\nAdditional Features\nLightweight racerback straps are easy to get on and off, and won't get in your way. \nFlattering V-neck silhouette. \nImported.\n\nSun Protection That Won't Wear Off\nOur high-performance fabric provides SPF", metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 3})



### Hard-coded examples


```python
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
```

### LLM-Generated examples


```python
from langchain.evaluation.qa import QAGenerateChain

```


```python
example_gen_chain = QAGenerateChain.from_llm(llm)
```


```python
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
```


```python
new_examples[0]
```




    {'query': "What is the weight of the Women's Campside Oxfords per pair?",
     'answer': "The Women's Campside Oxfords weigh approximately 1 lb. 1 oz. per pair."}




```python
data[0]
```




    Document(lc_kwargs={'page_content': ": 0\nname: Women's Campside Oxfords\ndescription: This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries.", 'metadata': {'source': 'OutdoorClothingCatalog_1000.csv', 'row': 0}}, page_content=": 0\nname: Women's Campside Oxfords\ndescription: This ultracomfortable lace-to-toe Oxford boasts a super-soft canvas, thick cushioning, and quality construction for a broken-in feel from the first time you put them on. \n\nSize & Fit: Order regular shoe size. For half sizes not offered, order up to next whole size. \n\nSpecs: Approx. weight: 1 lb.1 oz. per pair. \n\nConstruction: Soft canvas material for a broken-in feel and look. Comfortable EVA innersole with Cleansport NXT® antimicrobial odor control. Vintage hunt, fish and camping motif on innersole. Moderate arch contour of innersole. EVA foam midsole for cushioning and support. Chain-tread-inspired molded rubber outsole with modified chain-tread pattern. Imported. \n\nQuestions? Please contact us for any inquiries.", metadata={'source': 'OutdoorClothingCatalog_1000.csv', 'row': 0})



### Combine examples


```python
examples += new_examples
```


```python
qa.run(examples[0]["query"])
```

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    




    "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product."



## Manual Evaluation


```python
import langchain
langchain.debug = True
```


```python
qa.run(examples[0]["query"])
```

    [32;1m[1;3m[chain/start][0m [1m[1:chain:RetrievalQA] Entering Chain run with input:
    [0m{
      "query": "Do the Cozy Comfort Pullover Set        have side pockets?"
    }
    [32;1m[1;3m[chain/start][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain] Entering Chain run with input:
    [0m[inputs]
    [32;1m[1;3m[chain/start][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain] Entering Chain run with input:
    [0m{
      "question": "Do the Cozy Comfort Pullover Set        have side pockets?",
      "context": "Additional Features: Three-layer shell delivers waterproof protection. Brand new TEK O2 technology provides enhanced breathability. Interior gaiters keep out rain and snow. Full side zips for easy on/off over boots. Two zippered hand pockets. Thigh pocket. Imported.\n\n – Official Supplier to the U.S. Ski Team\nTHEIR WILL<<<<>>>>>: 5\nname: Smooth Comfort Check Shirt, Slightly Fitted\ndescription: Our men's slightly fitted check shirt is the perfect choice for your wardrobe! Customers love how it fits right out of the dryer. Size & Fit: Slightly Fitted, Relaxed through the chest and sleeve with a slightly slimmer waist. Fabric & Care: 100% cotton poplin, with wrinkle-free performance that won't wash out. Our innovative TrueCool® fabric wicks moisture away from your skin and helps it dry quickly. Additional Features: Traditional styling with a button-down collar and a single patch pocket. Imported.<<<<>>>>>: 4\nname: EcoFlex 3L Storm Pants\ndescription: Our new TEK O2 technology makes our four-season waterproof pants even more breathable. It's guaranteed to keep you dry and comfortable – whatever the activity and whatever the weather. Size & Fit: Slightly Fitted through hip and thigh. \n\nWhy We Love It: Our state-of-the-art TEK O2 technology offers the most breathability we've ever tested. Great as ski pants, they're ideal for a variety of outdoor activities year-round. Plus, they're loaded with features outdoor enthusiasts appreciate, including weather-blocking gaiters and handy side zips. Air In. Water Out. See how our air-permeable TEK O2 technology keeps you dry and comfortable. \n\nFabric & Care: 100% nylon, exclusive of trim. Machine wash and dry.<<<<>>>>>: 2\nname: Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece\ndescription: She'll love the bright colors, ruffles and exclusive whimsical prints of this toddler's two-piece swimsuit! Our four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. Machine wash and line dry for best results. Imported."
    }
    [32;1m[1;3m[llm/start][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain > 4:llm:AzureChatOpenAI] Entering LLM run with input:
    [0m{
      "prompts": [
        "System: Use the following pieces of context to answer the users question. \nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n----------------\nAdditional Features: Three-layer shell delivers waterproof protection. Brand new TEK O2 technology provides enhanced breathability. Interior gaiters keep out rain and snow. Full side zips for easy on/off over boots. Two zippered hand pockets. Thigh pocket. Imported.\n\n – Official Supplier to the U.S. Ski Team\nTHEIR WILL<<<<>>>>>: 5\nname: Smooth Comfort Check Shirt, Slightly Fitted\ndescription: Our men's slightly fitted check shirt is the perfect choice for your wardrobe! Customers love how it fits right out of the dryer. Size & Fit: Slightly Fitted, Relaxed through the chest and sleeve with a slightly slimmer waist. Fabric & Care: 100% cotton poplin, with wrinkle-free performance that won't wash out. Our innovative TrueCool® fabric wicks moisture away from your skin and helps it dry quickly. Additional Features: Traditional styling with a button-down collar and a single patch pocket. Imported.<<<<>>>>>: 4\nname: EcoFlex 3L Storm Pants\ndescription: Our new TEK O2 technology makes our four-season waterproof pants even more breathable. It's guaranteed to keep you dry and comfortable – whatever the activity and whatever the weather. Size & Fit: Slightly Fitted through hip and thigh. \n\nWhy We Love It: Our state-of-the-art TEK O2 technology offers the most breathability we've ever tested. Great as ski pants, they're ideal for a variety of outdoor activities year-round. Plus, they're loaded with features outdoor enthusiasts appreciate, including weather-blocking gaiters and handy side zips. Air In. Water Out. See how our air-permeable TEK O2 technology keeps you dry and comfortable. \n\nFabric & Care: 100% nylon, exclusive of trim. Machine wash and dry.<<<<>>>>>: 2\nname: Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece\ndescription: She'll love the bright colors, ruffles and exclusive whimsical prints of this toddler's two-piece swimsuit! Our four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. Machine wash and line dry for best results. Imported.\nHuman: Do the Cozy Comfort Pullover Set        have side pockets?"
      ]
    }
    [36;1m[1;3m[llm/end][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain > 4:llm:AzureChatOpenAI] [1.13s] Exiting LLM run with output:
    [0m{
      "generations": [
        [
          {
            "text": "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product.",
            "generation_info": null,
            "message": {
              "content": "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product.",
              "additional_kwargs": {},
              "example": false
            }
          }
        ]
      ],
      "llm_output": {
        "token_usage": {
          "completion_tokens": 29,
          "prompt_tokens": 561,
          "total_tokens": 590
        },
        "model_name": "gpt-35-turbo"
      },
      "run": null
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain > 3:chain:LLMChain] [1.13s] Exiting Chain run with output:
    [0m{
      "text": "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product."
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:RetrievalQA > 2:chain:StuffDocumentsChain] [1.13s] Exiting Chain run with output:
    [0m{
      "output_text": "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product."
    }
    [36;1m[1;3m[chain/end][0m [1m[1:chain:RetrievalQA] [1.57s] Exiting Chain run with output:
    [0m{
      "result": "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product."
    }
    




    "I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product."




```python
# Turn off the debug mode
langchain.debug = False
```

## LLM assisted evaluation


```python
predictions = qa.apply(examples)
```

    Error in on_chain_start callback: 'name'
    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    

    Error in on_chain_start callback: 'name'
    

    
    [1m> Finished chain.[0m
    
    [1m> Finished chain.[0m
    


```python
from langchain.evaluation.qa import QAEvalChain
```


```python
eval_chain = QAEvalChain.from_llm(llm)
```


```python
graded_outputs = eval_chain.evaluate(examples, predictions)
```


```python
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```

    Example 0:
    Question: Do the Cozy Comfort Pullover Set        have side pockets?
    Real Answer: Yes
    Predicted Answer: I'm sorry, but I don't have any information about the Cozy Comfort Pullover Set. The context provided does not mention this product.
    Predicted Grade: INCORRECT
    
    Example 1:
    Question: What collection is the Ultra-Lofty         850 Stretch Down Hooded Jacket from?
    Real Answer: The DownTek collection
    Predicted Answer: There is no information provided about the Ultra-Lofty 850 Stretch Down Hooded Jacket's collection.
    Predicted Grade: INCORRECT
    
    Example 2:
    Question: What is the weight of the Women's Campside Oxfords per pair?
    Real Answer: The Women's Campside Oxfords weigh approximately 1 lb. 1 oz. per pair.
    Predicted Answer: The Women's Campside Oxfords weigh approximately 1 lb. 1 oz. per pair.
    Predicted Grade: CORRECT
    
    Example 3:
    Question: What are the dimensions of the small and medium Recycled Waterhog Dog Mat?
    Real Answer: The small Recycled Waterhog Dog Mat has dimensions of 18" x 28" and the medium has dimensions of 22.5" x 34.5".
    Predicted Answer: The small Recycled Waterhog Dog Mat has dimensions of 18" x 28" and the medium size has dimensions of 22.5" x 34.5".
    Predicted Grade: CORRECT
    
    Example 4:
    Question: What are the features of the Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece?
    Real Answer: The Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece features bright colors, ruffles, and exclusive whimsical prints. The four-way-stretch and chlorine-resistant fabric keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. It can be machine washed and line dried for best results.
    Predicted Answer: The Infant and Toddler Girls' Coastal Chill Swimsuit, Two-Piece features bright colors, ruffles, and exclusive whimsical prints. The swimsuit is made of four-way-stretch and chlorine-resistant fabric that keeps its shape and resists snags. The UPF 50+ rated fabric provides the highest rated sun protection possible, blocking 98% of the sun's harmful rays. The crossover no-slip straps and fully lined bottom ensure a secure fit and maximum coverage. The swimsuit can be machine washed and line dried for best results.
    Predicted Grade: CORRECT
    
    Example 5:
    Question: What is the fabric composition of the Refresh Swimwear, V-Neck Tankini Contrasts?
    Real Answer: The Refresh Swimwear, V-Neck Tankini Contrasts is made of 82% recycled nylon and 18% Lycra® spandex for the body, and 90% recycled nylon and 10% Lycra® spandex for the lining.
    Predicted Answer: The Refresh Swimwear, V-Neck Tankini Contrasts is made of 82% recycled nylon with 18% Lycra® spandex for the body and 90% recycled nylon with 10% Lycra® spandex for the lining.
    Predicted Grade: CORRECT
    
    Example 6:
    Question: What is the new technology used in the EcoFlex 3L Storm Pants and what is its benefit?
    Real Answer: The new technology used in the EcoFlex 3L Storm Pants is TEK O2 technology, which makes the pants more breathable. Its benefit is that it keeps the wearer dry and comfortable in any activity and weather.
    Predicted Answer: The new technology used in the EcoFlex 3L Storm Pants is called TEK O2 technology, which offers enhanced breathability. This means that the pants are more comfortable to wear and will keep you dry and comfortable during outdoor activities, no matter the weather.
    Predicted Grade: CORRECT
    
    


```python

```
