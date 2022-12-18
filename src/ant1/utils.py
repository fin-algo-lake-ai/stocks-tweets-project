from random import choice, choices

from jinja2 import Template


def print_sample(data, toxicity_level, n_samples=5):
    return "\n\n".join(choices(data[toxicity_level], k=n_samples))


def show_example(data, ratings=(0, 0, 1, 1, 2, 2, 3, 3, 4, 5), max_len=400):
    """
    Shows some random examples for each toxicity rating in `ratings`. Total numbers of shown comments
    is equal to `ratings` length.
    :param data: toxic comments data
    :param ratings: ratings to show
    :param max_len: maximum characters to show in a comment
    :return:
    """
    my_template = Template("""
    <!DOCTYPE html>
    <html>
    <table>
    <tr> <th>Toxicity</th> <th>Comment</th>
    {% for item in items %}
    <TR>
       <TD class="c1" style="text-align:center; font-size:large" title="Toxicity Rating: {{item.rating}}">{{item.emoji}}</TD>
       <TD class="c2" style="font-size:large">{{item.text}}</TD>
    </TR>
    {% endfor %}
    </table>
    </html>""")
    rating2emoji = {0: "&#128519;",
                    1: "&#128528;",
                    2: "&#128551;",
                    3: "&#128565;",
                    4: "&#128557;",
                    5: "&#128561;"}
    items = []
    for rating in ratings:
        text = choice(data[rating])
        if len(text) > max_len:
            text = text[:max_len] + "..."
        items.append({"text": text,
                      "rating": rating,
                      "emoji": rating2emoji[rating]})
    return my_template.render(items=items)
