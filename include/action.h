#ifndef ACTION_H
#define ACTION_H

struct Action {
    int from; // Source square index (0-89)
    int to;   // Destination square index (0-89)

    bool operator==(const Action& other) const {
        return from == other.from && to == other.to;
    }
};

#endif // ACTION_H
