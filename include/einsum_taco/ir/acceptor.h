//
// Created by Alexandra Dima on 04.02.2022.
//

#ifndef EINSUM_TACO_ACCEPTOR_H
#define EINSUM_TACO_ACCEPTOR_H

#endif //EINSUM_TACO_ACCEPTOR_H


#include<string>
#include<vector>
#include<einsum_taco/ir/type.h>
#include<einsum_taco/base/assert.h>
#include<set>
#include<map>
#include<memory>


namespace einsum {

    class IRContext;
    class IRVisitor;

    struct IR : std::enable_shared_from_this<IR> {

        // TODO: make this a visitor instead
        virtual std::string dump() const = 0;

        virtual void accept(IRVisitor* v) = 0;

        std::string class_name() const;

        template<typename T, typename ... Types >
        static std::shared_ptr<T> make(Types... args) {
            return std::make_shared<T>(args...);
        }

        template<typename T, typename ... Types >
        static std::vector<std::shared_ptr<T>> make_vec(Types... args) {
            std::vector<std::shared_ptr<T>> v = {args...};
            return v;
        }

        virtual ~IR() = default;
    };

    template<typename T, typename parent = IR, typename... mixins>
    struct Acceptor : parent, mixins... {
        using Base = Acceptor<T, parent, mixins...>;
        using parent :: parent;
       ~Acceptor() override = default;
        void accept(IRVisitor* v) override;
    };
}