from transformers import AutoTokenizer, T5ForConditionalGeneration

print('loading tokenizer')
tokenizer = AutoTokenizer.from_pretrained('thegoodfellas/tgf-ptbr-flan-t5-base-2th', use_auth_token=True)

print('loading model')
model = T5ForConditionalGeneration.from_pretrained('experiments/harem-tgf-ptbr-flan-t5-base-2th-ep100-len256-b8-train,test/episode0/')
model.eval()

device = 'cuda'

model.to(device)

sents = [
    'Estamos aqui na Feira Internacional de IA com o senhor Paulo para acompanhar o evento de 2023.',
    'A nova sede do Banco do Brasil fica localizada na Rua das Palmeiras, 137. É a primeira mudança desde que o presidente atual, Dr. João, entrou',
    'O novo endereço do Maracanã agora é na Av. Bandeirantes, no bairro da Lapa. A presidente, Maria, fala ao vivo hoje'
]

for s in sents:
    inputs = tokenizer(
        s,
        max_length=512,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
    )

    print('predicting')
    predictions = model.generate(
        inputs['input_ids'].to(device),
        max_length=512,
        num_beams=8,
        repetition_penalty=0.6
    )

    gen = tokenizer.decode(predictions[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(f'### Original: {s}')
    print(f'### Predict: {gen}')

