import re
import torch
from transformers import BertTokenizer

MAX_LEN = 400

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer_chi = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer_chi.add_tokens(["[CLS2]", "[CLS3]"])


def text_preprocessing(text):
    # text = re.sub(r'(@.*?)[\s]', ' ', text)
    # text = re.sub(r'[0-9]+', '', text)
    # text = re.sub(r'\s([@][\w_-]+)', '', text).strip()
    # text = re.sub(r'&amp;', '&', text)
    # text = re.sub(r'\s+', ' ', text).strip()
    # text = text.replace("#", " ")
    # encoded_string = text.encode("ascii", "ignore")
    # decode_string = encoded_string.decode()
    text = '[CLS2] [CLS3]' + text
    
    return text


def preprocessing_for_bert(data):
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        
        encoded_sent = tokenizer_chi.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            padding="max_length",         # Pad sentence to max length
            # return_tensors='pt',             # Return PyTorch tensor
            truncation=True,
            return_attention_mask=True      # Return attention mask
        )
        
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def detokenizer(input_ids):
    return tokenizer_chi.convert_ids_to_tokens(input_ids)

# if __name__=='__main__':
    
#     sent = '冒煙的喬 高雄忠孝店 108.\n菜單改了, 菜樣有增減, 價格也有調整, 風味和往常也有別, 如玉米拼盤, 之前上面的蔬菜舖很多, 現在只有很薄的一些些..\n110.2.27\n這家店是我婚前吃到現在小孩都成人了, 這次隔了好久之後才來, 從進門時的招牌燈光就顯得微弱, 似乎在訴說這家店的年華, 進門看服務人員換了, 擺設依舊, 但環境不如以前,\n從帶位就不舒服, 樓上座位其實空位不少, 偏偏被指定坐到角落防火器邊, 這地方的斑駁牆面看了不舒服. 重點是有"角落生物--小強"一直在四周遊走, 用餐過程小孩一直在躲, 無法讓人安心用餐..\n期間向服務人員反應, 如果旁邊沒人坐, 可以換位嗎, 回覆說怕等下有客人來. 直到離開時, 並沒有, 可以跟老闆說的是, 員工訓練的好是可以帶來客源, 反之, 再忠實的顧客也留不住.\n喜歡吃的玉米餅拼盤. 上面的舖菜切的很粗, 功法很差, 鋪陳更少, 有的是堆疊很亂且都是小小片的玉米餅, 如果只是要吃餅, 給我一袋玉米餅吃就行了, 不需要用拼盤價賣...\n再者, 端上來的菜, 嚇一跳, 怎麼色澤這麼黑, 是鍋子的問題還是油質的問題.\n吃飯時, 旁邊是整理區, 一直有盤子的碰撞聲, 服務人員的交談聲,\n用餐品質真的很差...\n前後一個小時就走人了.. 5/24\n義式炒綜合蘑菇::蘑菇不會因為加了一些調味而失去應有的味道，吃起來口感有一點接近台式炒菇類的感覺，覺得適合配飯。\n金黃花枝酥條:花枝條每一根都粗，不是細細一條這樣，炸的口感也很剛好，可以搭配胡椒鹽或是奶焗醬，都很好吃。\n玉米餅總匯拼盤:玉米餅外觀像是多力多滋那樣的形狀，鋪在最底層，上面放滿生菜以及些許辣椒，搭配莎莎醬也很不錯。\n新鮮橄欖油墨魚麵:墨魚麵條軟硬口感剛好，加上蒜片、花枝等一起煮，也是很好吃。\n酥烤德國豬腳:德國豬腳的外皮烤的外酥內軟，又帶點嚼勁，豬腳肉本身也不會太軟，搭配酸菜更好吃。\n義大利香料烤全雞:烤雞本身應該有用香料醃過，肉質軟嫩，底下還有鋪一層類似香料植物以及奶焗白醬的東西，吃起來有點特別。\n蔓越莓汁:雖然是點無糖，店家也有說想加糖可以再向他們反映，只是無糖喝起來很像是紅茶的味道，不太像是蔓越莓汁。'
#     sent = '[CLS2] [CLS3]' + sent
    
#     tokenizer_chi.add_tokens(["[CLS2]", "[CLS3]"])
#     encoded_sent = tokenizer_chi.encode_plus(
#         text=sent,
#         add_special_tokens=True,
#         max_length=MAX_LEN,
#         padding="max_length",
#         truncation=True,
#         return_attention_mask=True,
#     )
    
#     input_ids = encoded_sent.get('input_ids')
    
#     print(input_ids)
#     print(tokenizer_chi.convert_ids_to_tokens(input_ids))
    