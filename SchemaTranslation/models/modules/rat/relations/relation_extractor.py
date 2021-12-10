import torch

from models.modules.rat.relations.relations_builder_value_type4 import build_relations_target_to_context


class RelationExtractor:
    def __init__(self, include_types, include_structure, pad_id, eos_id, context_sep_id, item_sep_id,
                 language_token_src_id):

        self.include_types = include_types
        self.include_structure = include_structure

        self.pad_id = pad_id
        self.eos_id = eos_id
        self.context_sep_id = context_sep_id
        self.item_sep_id = item_sep_id
        self.language_token_src_id = language_token_src_id

        assert include_structure or include_types, "at least one of the values from include_types or " \
                                                   "include_structure must be True "
        if not include_structure:  # types-aware
            self.relations = {
                "default": 0,
                "self_target": 1,
                "self_siblings": 2,
                "self_values": 3,
                # "self_defaulted": 4,
            }
        elif not include_types:  # structure-aware
            self.relations = {
                "default": 0,
                "sibling_headers": 1,
                "related_values": 2,
                "context_sep": 3,
                "language_token": 4,
                "item_sep": 5,
                "self": 6,
                "eos": 7}
        else:  # structure-aware + type-aware
            self.relations = {
                "default": 0,
                "sibling_headers": 1,
                "related_values": 2,
                "context_sep": 3,
                "language_token": 4,
                "item_sep": 5,
                "self_target": 6,
                "self_siblings": 7,
                "self_values": 8,
                # "self_defaulted": 9,
                "eos": 9,
            }
        self.num_relations = len(self.relations)

    def fill_self_relations(self, index, seperator_masks, context_sep_indicies):
        if self.include_types:
            type_aware_self_relations = [self.relations["self_target"], self.relations["self_siblings"],
                                         self.relations["self_values"]]

            def fill_type_aware_relations(mask, pos):
                if pos <= context_sep_indicies[0]:
                    self.relation_matrix[index].masked_fill_(mask, type_aware_self_relations[0])
                elif len(context_sep_indicies) == 1:  # the rest can only be siblings or paddings
                    self.relation_matrix[index].masked_fill_(mask, type_aware_self_relations[1])  # siblings
                else:  # the rest can be siblings/ values/ or paddings
                    if context_sep_indicies[0] < pos <= context_sep_indicies[1]:
                        self.relation_matrix[index].masked_fill_(mask, type_aware_self_relations[1])  # siblings
                    else:
                        self.relation_matrix[index].masked_fill_(mask,
                                                                 type_aware_self_relations[-1])  # values or paddings

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
            if not self.include_types:  # structure-aware only
                self.relation_matrix[index].masked_fill_(mask, self.relations["self"])
            else:  # type-aware
                fill_type_aware_relations(mask, sep)

        for i in range(sliding_mask.size(0)):
            if not self.include_types:
                self.relation_matrix[index][i][i] = self.relations["self"]
            else:
                mask = torch.zeros(seperator_masks.size(-1),
                                   seperator_masks.size(-1),
                                   dtype=torch.bool).to(seperator_masks.device)
                mask[i][i] = True
                fill_type_aware_relations(mask, i)

            # # 05-13-19-00
            # if i in separator_indices:  # all sep including lang-tok
            #     self.relation_matrix[index][i][i] = self.relations["default"]

            # # 05-14-15-30 currently best
            # if i in separator_indices:  # all sep except for lang-tok
            #     if not i ==0:
            #         self.relation_matrix[index][i][i] = self.relations["default"]

            # 05-16-16-13
            if i in context_sep_indicies:  # all context-sep plus lang-tok
                    self.relation_matrix[index][i][i] = self.relations["default"]
            i=0
            self.relation_matrix[index][i][i] = self.relations["default"]

            # # 05-14-19-25
            # if i in context_sep_indicies:  # all context-sep
            #         self.relation_matrix[index][i][i] = self.relations["default"]

    def build_relations(self, batch_tokens):
        shape = batch_tokens.shape
        self.relation_matrix = torch.zeros(shape[0], shape[1], shape[1]).cuda()
        for i, tokens in enumerate(batch_tokens):
            assert int(tokens[0]) == self.language_token_src_id
            context_sep_id_all = tokens.eq(self.context_sep_id).nonzero(as_tuple=True)[0]
            context_sep_id = int(context_sep_id_all) if context_sep_id_all.size(0) == 1 else int(context_sep_id_all[0])
            sep_i = tokens.eq(self.item_sep_id)
            sep_c = tokens.eq(self.context_sep_id)
            sep_mask = tokens.eq(self.language_token_src_id) | sep_i | sep_c
            sep_mask[-1] = True

            if not self.include_structure:  # type-aware only
                self.fill_self_relations(i, sep_mask, context_sep_id_all)
            else:
                target_ids = [x for x in range(1, context_sep_id)]  # zero for language token
                if len(context_sep_id_all) == 1:  # schema only
                    # siblings headers
                    self.relation_matrix[i][target_ids, :] = self.relations["sibling_headers"]
                    self.relation_matrix[i][:, target_ids] = self.relations["sibling_headers"]
                elif len(context_sep_id_all) == 2:  # schema and value
                    last_context_sep = int(context_sep_id_all[-1])
                    # siblings headers
                    self.relation_matrix[i][target_ids, :last_context_sep] = self.relations["sibling_headers"]
                    self.relation_matrix[i][:last_context_sep, target_ids] = self.relations["sibling_headers"]
                    # related values
                    self.relation_matrix[i][target_ids, last_context_sep:] = self.relations["related_values"]
                    self.relation_matrix[i][last_context_sep:, target_ids] = self.relations["related_values"]
                else:
                    raise Exception

                # fill language tokens relations
                self.relation_matrix[i][[0], :] = self.relations["language_token"]
                self.relation_matrix[i][:, [0]] = self.relations["language_token"]

                # fill item_sep relations
                sep_ids = sep_i.nonzero(as_tuple=True)[0]
                self.relation_matrix[i][sep_ids, :] = self.relations["item_sep"]
                self.relation_matrix[i][:, sep_ids] = self.relations["item_sep"]

                # fill context_sep relations
                self.relation_matrix[i][context_sep_id_all, :] = self.relations["context_sep"]
                self.relation_matrix[i][:, context_sep_id_all] = self.relations["context_sep"]

                # fill self relations
                self.fill_self_relations(i, sep_mask, context_sep_id_all)

                # fill eos relations
                eos_ids = tokens.eq(self.eos_id).nonzero(as_tuple=True)[0]
                self.relation_matrix[i][[eos_ids], :] = self.relations["eos"]
                self.relation_matrix[i][:, [eos_ids]] = self.relations["eos"]

            # fill paddings
            # mask from the input of rat can be not quite reliable
            pad_ids = tokens.eq(self.pad_id).nonzero(as_tuple=True)[0]
            if len(pad_ids):
                self.relation_matrix[i][pad_ids, :] = self.relations["default"]
                self.relation_matrix[i][:, pad_ids] = self.relations["default"]

        return self.relation_matrix


if __name__ == '__main__':
    extractor = RelationExtractor(
        # include_types=False,
        include_types=True,
        # include_structure=True,
        include_structure=False,
        pad_id=1,
        eos_id=2,
        context_sep_id=128113,
        item_sep_id=128112,
        language_token_src_id=128022)
    print(extractor.num_relations)
    src_tokens = torch.tensor([
        [128022, 15937, 8, 39095, 33970, 34760, 404, 237, 451,
         128113, 103156, 103156, 128112, 291, 12812, 128112, 41252, 291, 4182,
         1521, 128112, 83282, 47330, 128112, 960, 20973, 2],
        [128022, 15937, 8, 39095, 33970, 34760, 404, 237, 451,
         128113, 103156, 103156, 128112, 291, 12812, 128112, 41252, 4182,
         1521, 128112, 83282, 47330, 128112, 960, 2, 1, 1],
        [128022, 7541, 178, 397, 128113, 103156, 42539, 128112, 26943, 128113, 82,
         3531, 128112, 2, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]).cuda()
    for r in extractor.build_relations(src_tokens):
        print(r.shape)
        print("relations:")
        for line in r.tolist():
            print([int(i) for i in line])
        print("flatten")
        types = torch.max(r, dim=1).values.type_as(src_tokens)
        print(types.tolist())
