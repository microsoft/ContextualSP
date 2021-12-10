import torch


def get_self_mask(seperator_masks):
    separator_indices = seperator_masks.nonzero(as_tuple=False).view(-1)
    sliding_mask = torch.zeros(seperator_masks.size(-1),
                               seperator_masks.size(-1),
                               dtype=torch.bool).to(seperator_masks.device)
    last_sep = torch.tensor(-1).to(seperator_masks.device)
    for i, sep in enumerate(separator_indices):
        vertical_mask = torch.arange(seperator_masks.size(-1)).unsqueeze(0).to(seperator_masks.device)
        vertical_mask = vertical_mask.repeat((seperator_masks.size(-1), 1))
        vertical_mask = vertical_mask.ge(last_sep + 1) & vertical_mask.le(sep - 1)

        horizontal_mask = torch.arange(seperator_masks.size(-1)).unsqueeze(0).to(seperator_masks.device)
        horizontal_mask = horizontal_mask.repeat((seperator_masks.size(-1), 1)).T
        horizontal_mask = horizontal_mask.ge(last_sep + 1) & horizontal_mask.le(sep - 1)

        last_sep = sep

        mask = horizontal_mask & vertical_mask
        assert sliding_mask.shape == mask.shape
        sliding_mask = torch.logical_xor(sliding_mask, mask)

    for i in range(sliding_mask.size(0)):
        sliding_mask[i][i] = True
    return sliding_mask


def build_relations_target_to_context(batch_tokens,relation_tokens,relation_map):
    item_sep,context_sep,eos,pad,language_token_en = relation_tokens["item_sep"], \
                                                     relation_tokens["context_sep"], \
                                                     relation_tokens["eos"],\
                                                     relation_tokens["pad"],\
                                                     relation_tokens["language_token_en"]
    shape = batch_tokens.shape
    relations = torch.zeros(shape[0], shape[1], shape[1]).cuda()
    for i, tokens in enumerate(batch_tokens):
        if language_token_en:
            assert int(tokens[0]) == language_token_en
        context_sep_id_all = tokens.eq(context_sep).nonzero(as_tuple=True)[0]
        context_sep_id = int(context_sep_id_all) if context_sep_id_all.size(0) == 1 else int(context_sep_id_all[0])
        if language_token_en:
            target_ids = [x for x in range(1, context_sep_id)]  # zero for language token
        else:
            target_ids = [x for x in range(0, context_sep_id)]

        if len(context_sep_id_all) == 1:  # schema only
            # siblings headers
            relations[i][target_ids, :] = relation_map["siblings_header"]
            relations[i][:, target_ids] = relation_map["siblings_header"]
        elif len(context_sep_id_all) == 2:  # schema and value
            last_context_sep = int(context_sep_id_all[-1])
            # siblings headers
            relations[i][target_ids, :last_context_sep] = relation_map["siblings_header"]
            relations[i][:last_context_sep, target_ids] = relation_map["siblings_header"]
            # related values
            relations[i][target_ids, last_context_sep:] = relation_map["related_value"]
            relations[i][last_context_sep:, target_ids] = relation_map["related_value"]
        else:
            raise Exception

        # fill language tokens relations
        if language_token_en:
            relations[i][[0], :] = relation_map["language_token"]
            relations[i][:, [0]] = relation_map["language_token"]

        # fill item_sep relations
        sep_i = tokens.eq(item_sep)
        sep_ids = sep_i.nonzero(as_tuple=True)[0]
        relations[i][sep_ids, :] = relation_map["item_sep"]
        relations[i][:, sep_ids] = relation_map["item_sep"]

        # fill context_sep relations
        sep_c = tokens.eq(context_sep)
        relations[i][context_sep_id_all, :] = relation_map["context_sep"]
        relations[i][:, context_sep_id_all] = relation_map["context_sep"]

        # fill self relations
        if language_token_en:
            sep_mask = tokens.eq(language_token_en) | sep_i | sep_c
        else:
            sep_mask = sep_i | sep_c
        sep_mask[-1] = True
        mask = get_self_mask(sep_mask).cuda()
        relations[i].masked_fill_(mask, relation_map["self"])

        # fill eos relations
        eos_ids = tokens.eq(eos).nonzero(as_tuple=True)[0]
        relations[i][[eos_ids], :] = relation_map["eos"]
        relations[i][:, [eos_ids]] = relation_map["eos"]

        # fill paddings
        # mask from the input of rat can be not quite reliable
        pad_ids = tokens.eq(pad).nonzero(as_tuple=True)[0]
        if len(pad_ids):
            relations[i][pad_ids, :] = relation_map["default"]
            relations[i][:, pad_ids] = relation_map["default"]

    return relations
def get_type_mask(index,relation_matrix):
    size=relation_matrix.size(-1)
    vertical_mask = torch.arange(size).unsqueeze(0).type_as(relation_matrix)
    vertical_mask = vertical_mask.repeat((size, 1))
    vertical_mask = vertical_mask.le(index)
    mask = (vertical_mask & vertical_mask.T)
    return mask

def relation_to_types(relation,eos,context_sep=3):
    context_sep_indices = relation[0].eq(context_sep).nonzero(as_tuple=True)[0]
    first_context_sep_index = context_sep_indices[0]
    first_sep_mask = get_type_mask(first_context_sep_index,relation)

    if len(context_sep_indices)==2:
        second_context_sep_index = context_sep_indices[1]
        second_sep_mask = get_type_mask(second_context_sep_index,relation)
    else:
        second_sep_mask = torch.zeros_like(first_sep_mask).type_as(first_sep_mask)
    eos_index=relation[0].eq(eos).nonzero(as_tuple=True)[0][0]
    last_sep_mask = get_type_mask(eos_index,relation)

    types=torch.zeros_like(relation)
    types.masked_fill_(last_sep_mask,3)
    types.masked_fill_(second_sep_mask,2)
    types.masked_fill_(first_sep_mask,1)
    return types.type_as(relation)
if __name__ == '__main__':
    # src_tokens = torch.tensor([
    #     [128022, 15937, 8, 39095, 33970, 34760, 404, 237, 451,
    #      128113, 103156, 128112, 291, 12812, 128112, 41252, 291, 4182,
    #      1521, 128112, 83282, 47330, 960, 20973, 128112, 83282, 117389,
    #      960, 20973, 128112, 3466, 960, 5170, 60929, 50, 128112,
    #      15937, 8, 39095, 33970, 34760, 404, 128112, 20416, 33970,
    #      34760, 404, 128112, 121066, 116464, 4, 138, 124366, 128112,
    #      75848, 404, 12, 55, 124366, 128112, 38080, 4513, 12,
    #      55, 124366, 128112, 2085, 2436, 404, 1246, 60163, 124366,
    #      128112, 83282, 47330, 960, 20973, 128112, 20416, 33970, 34760,
    #      404, 21738, 128112, 121066, 116464, 4, 138, 124366, 237,
    #      451, 128112, 75848, 404, 12, 55, 124366, 237, 451,
    #      128112, 38080, 4513, 12, 55, 124366, 237, 451, 128112,
    #      2085, 2436, 404, 1246, 60163, 124366, 237, 451, 128112,
    #      83282, 117389, 960, 20973, 128112, 15937, 8, 39095, 33970,
    #      34760, 404, 237, 339, 128112, 20416, 33970, 34760, 404,
    #      28539, 128112, 121066, 116464, 4, 138, 124366, 237, 339,
    #      128112, 75848, 404, 12, 55, 124366, 237, 339, 128112,
    #      38080, 4513, 12, 55, 124366, 237, 339, 128112, 2085,
    #      2436, 404, 1246, 60163, 124366, 237, 339, 2],
    #     [128022, 7541, 178, 397, 128113, 42539, 128112, 26943, 82,
    #      3531, 128112, 10709, 1463, 128112, 53012, 128112, 13939, 492,
    #      4760, 128112, 56820, 161, 128112, 56820, 168, 2, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1, 1,
    #      1, 1, 1, 1, 1, 1, 1, 1]
    # ]).cuda()
    src_tokens = torch.tensor([
        [128022, 15937, 8, 39095, 33970, 34760, 404, 237, 451,
         128113, 103156, 103156, 128112, 291, 12812, 128112, 41252, 291, 4182,
         1521, 128112, 83282, 47330, 128112, 960, 20973, 2],
        [128022, 7541, 178, 397, 128113, 103156, 42539, 128112, 26943, 128113, 82,
         3531, 128112, 2, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]).cuda()
    for r, t in zip(build_relations_target_to_context(src_tokens,
                            {"pad":1,
                            "eos":2,
                            "item_sep":128112,
                            "context_sep":128113,
                            "language_token_en": 128022
                                    }, relation_map=
    {"default": 0, "siblings_header": 1, "related_value": 2, "context_sep": 3,
     "language_token": 4,
     "item_sep": 5,
     "self": 6, "eos": 7}), src_tokens):
        # print(t.tolist())
        print(r.shape,t.shape)
        print("relations:")
        for line in r.tolist():
            print([int(i) for i in line])
        print("types:")
        for line in relation_to_types(r,7).tolist():
            print([int(i) for i in line])
    print("no lang_tok","-"*100)
    src_tokens = torch.tensor([
        [15937, 8, 39095, 33970, 34760, 404, 237, 451,
         128113, 103156, 103156, 128112, 291, 12812, 128112, 41252, 291, 4182,
         1521, 128112, 83282, 47330, 128112, 960, 20973, 2],
        [7541, 178, 397, 128113, 103156, 42539, 128112, 26943, 128113, 82,
         3531, 128112, 2, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]).cuda()
    for r, t in zip(build_relations_target_to_context(src_tokens, {"pad":1,"eos":2,"item_sep":128112,"context_sep":128113,"language_token_en": None}, relation_map=
    {"default": 0, "siblings_header": 1, "related_value": 2, "context_sep": 3,"item_sep": 4,"self": 5, "eos": 6}), src_tokens):
        print("relations:")
        for line in r.tolist():
            print([int(i) for i in line])
        print("types:")
        for line in relation_to_types(r,6).tolist():
            print([int(i) for i in line])