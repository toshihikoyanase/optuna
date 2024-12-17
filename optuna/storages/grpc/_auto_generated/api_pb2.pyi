"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
*
Optuna GRPC API
The following command generates the Python code from this file:
$ pip install mypy-protobuf==3.6.0 protobuf==5.28.1 grpcio==1.68.1 grpcio-tools==1.68.1
$ python -m grpc_tools.protoc \\
--proto_path=optuna/storages/grpc \\
--grpc_python_out=optuna/storages/grpc/_auto_generated \\
--python_out=optuna/storages/grpc/_auto_generated \\
--mypy_out=optuna/storages/grpc/_auto_generated \\
optuna/storages/grpc/api.proto
$ sed -i -e \\
"s/import api_pb2 as api__pb2/import optuna.storages.grpc._auto_generated.api_pb2 as api__pb2/g" \\
optuna/storages/grpc/_auto_generated/api_pb2_grpc.py
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _StudyDirection:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _StudyDirectionEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_StudyDirection.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    MINIMIZE: _StudyDirection.ValueType  # 0
    MAXIMIZE: _StudyDirection.ValueType  # 1

class StudyDirection(_StudyDirection, metaclass=_StudyDirectionEnumTypeWrapper):
    """*
    Study direction.
    """

MINIMIZE: StudyDirection.ValueType  # 0
MAXIMIZE: StudyDirection.ValueType  # 1
global___StudyDirection = StudyDirection

class _TrialState:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _TrialStateEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_TrialState.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    RUNNING: _TrialState.ValueType  # 0
    COMPLETE: _TrialState.ValueType  # 1
    PRUNED: _TrialState.ValueType  # 2
    FAIL: _TrialState.ValueType  # 3
    WAITING: _TrialState.ValueType  # 4

class TrialState(_TrialState, metaclass=_TrialStateEnumTypeWrapper):
    """*
    Trial state.
    """

RUNNING: TrialState.ValueType  # 0
COMPLETE: TrialState.ValueType  # 1
PRUNED: TrialState.ValueType  # 2
FAIL: TrialState.ValueType  # 3
WAITING: TrialState.ValueType  # 4
global___TrialState = TrialState

@typing.final
class CreateNewStudyRequest(google.protobuf.message.Message):
    """*
    ========================================
    Messages for Optuna storage service.
    ========================================

    *
    Request to create a new study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DIRECTIONS_FIELD_NUMBER: builtins.int
    STUDY_NAME_FIELD_NUMBER: builtins.int
    study_name: builtins.str
    @property
    def directions(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___StudyDirection.ValueType]: ...
    def __init__(
        self,
        *,
        directions: collections.abc.Iterable[global___StudyDirection.ValueType] | None = ...,
        study_name: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["directions", b"directions", "study_name", b"study_name"]) -> None: ...

global___CreateNewStudyRequest = CreateNewStudyRequest

@typing.final
class CreateNewStudyReply(google.protobuf.message.Message):
    """*
    Reply to create a new study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___CreateNewStudyReply = CreateNewStudyReply

@typing.final
class DeleteStudyRequest(google.protobuf.message.Message):
    """*
    Request to delete a study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___DeleteStudyRequest = DeleteStudyRequest

@typing.final
class DeleteStudyReply(google.protobuf.message.Message):
    """*
    Reply to delete a study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___DeleteStudyReply = DeleteStudyReply

@typing.final
class SetStudyUserAttributeRequest(google.protobuf.message.Message):
    """*
    Request to set a study's user attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    key: builtins.str
    value: builtins.str
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        key: builtins.str = ...,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["key", b"key", "study_id", b"study_id", "value", b"value"]) -> None: ...

global___SetStudyUserAttributeRequest = SetStudyUserAttributeRequest

@typing.final
class SetStudyUserAttributeReply(google.protobuf.message.Message):
    """*
    Reply to set a study's user attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetStudyUserAttributeReply = SetStudyUserAttributeReply

@typing.final
class SetStudySystemAttributeRequest(google.protobuf.message.Message):
    """*
    Request to set a study's system attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    key: builtins.str
    value: builtins.str
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        key: builtins.str = ...,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["key", b"key", "study_id", b"study_id", "value", b"value"]) -> None: ...

global___SetStudySystemAttributeRequest = SetStudySystemAttributeRequest

@typing.final
class SetStudySystemAttributeReply(google.protobuf.message.Message):
    """*
    Reply to set a study's system attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetStudySystemAttributeReply = SetStudySystemAttributeReply

@typing.final
class GetStudyIdFromNameRequest(google.protobuf.message.Message):
    """*
    Request to get a study id by its name.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_NAME_FIELD_NUMBER: builtins.int
    study_name: builtins.str
    def __init__(
        self,
        *,
        study_name: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_name", b"study_name"]) -> None: ...

global___GetStudyIdFromNameRequest = GetStudyIdFromNameRequest

@typing.final
class GetStudyIdFromNameReply(google.protobuf.message.Message):
    """*
    Reply to get a study id by its name.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___GetStudyIdFromNameReply = GetStudyIdFromNameReply

@typing.final
class GetStudyNameFromIdRequest(google.protobuf.message.Message):
    """*
    Request to get a study name by its id.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___GetStudyNameFromIdRequest = GetStudyNameFromIdRequest

@typing.final
class GetStudyNameFromIdReply(google.protobuf.message.Message):
    """*
    Reply to get a study name by its id.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_NAME_FIELD_NUMBER: builtins.int
    study_name: builtins.str
    def __init__(
        self,
        *,
        study_name: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_name", b"study_name"]) -> None: ...

global___GetStudyNameFromIdReply = GetStudyNameFromIdReply

@typing.final
class GetStudyDirectionsRequest(google.protobuf.message.Message):
    """*
    Request to get study directions.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___GetStudyDirectionsRequest = GetStudyDirectionsRequest

@typing.final
class GetStudyDirectionsReply(google.protobuf.message.Message):
    """*
    Reply to get study directions.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DIRECTIONS_FIELD_NUMBER: builtins.int
    @property
    def directions(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___StudyDirection.ValueType]: ...
    def __init__(
        self,
        *,
        directions: collections.abc.Iterable[global___StudyDirection.ValueType] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["directions", b"directions"]) -> None: ...

global___GetStudyDirectionsReply = GetStudyDirectionsReply

@typing.final
class GetStudyUserAttributesRequest(google.protobuf.message.Message):
    """*
    Request to get study user attributes.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___GetStudyUserAttributesRequest = GetStudyUserAttributesRequest

@typing.final
class GetStudyUserAttributesReply(google.protobuf.message.Message):
    """*
    Reply to get study user attributes.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class UserAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    USER_ATTRIBUTES_FIELD_NUMBER: builtins.int
    @property
    def user_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    def __init__(
        self,
        *,
        user_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["user_attributes", b"user_attributes"]) -> None: ...

global___GetStudyUserAttributesReply = GetStudyUserAttributesReply

@typing.final
class GetStudySystemAttributesRequest(google.protobuf.message.Message):
    """*
    Request to get study system attributes.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id"]) -> None: ...

global___GetStudySystemAttributesRequest = GetStudySystemAttributesRequest

@typing.final
class GetStudySystemAttributesReply(google.protobuf.message.Message):
    """*
    Reply to get study system attributes.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class SystemAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    SYSTEM_ATTRIBUTES_FIELD_NUMBER: builtins.int
    @property
    def system_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    def __init__(
        self,
        *,
        system_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["system_attributes", b"system_attributes"]) -> None: ...

global___GetStudySystemAttributesReply = GetStudySystemAttributesReply

@typing.final
class GetAllStudiesRequest(google.protobuf.message.Message):
    """*
    Request to get all studies.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___GetAllStudiesRequest = GetAllStudiesRequest

@typing.final
class GetAllStudiesReply(google.protobuf.message.Message):
    """*
    Reply to get all studies.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDIES_FIELD_NUMBER: builtins.int
    @property
    def studies(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Study]: ...
    def __init__(
        self,
        *,
        studies: collections.abc.Iterable[global___Study] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["studies", b"studies"]) -> None: ...

global___GetAllStudiesReply = GetAllStudiesReply

@typing.final
class CreateNewTrialRequest(google.protobuf.message.Message):
    """*
    Request to create a new trial.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    TEMPLATE_TRIAL_FIELD_NUMBER: builtins.int
    TEMPLATE_TRIAL_IS_NONE_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    template_trial_is_none: builtins.bool
    @property
    def template_trial(self) -> global___Trial: ...
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        template_trial: global___Trial | None = ...,
        template_trial_is_none: builtins.bool = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["template_trial", b"template_trial"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id", "template_trial", b"template_trial", "template_trial_is_none", b"template_trial_is_none"]) -> None: ...

global___CreateNewTrialRequest = CreateNewTrialRequest

@typing.final
class CreateNewTrialReply(google.protobuf.message.Message):
    """*
    Reply to create a new trial.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trial_id", b"trial_id"]) -> None: ...

global___CreateNewTrialReply = CreateNewTrialReply

@typing.final
class SetTrialParameterRequest(google.protobuf.message.Message):
    """*
    Request to set a trial parameter.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    PARAM_NAME_FIELD_NUMBER: builtins.int
    PARAM_VALUE_INTERNAL_FIELD_NUMBER: builtins.int
    DISTRIBUTION_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    param_name: builtins.str
    param_value_internal: builtins.float
    distribution: builtins.str
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        param_name: builtins.str = ...,
        param_value_internal: builtins.float = ...,
        distribution: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["distribution", b"distribution", "param_name", b"param_name", "param_value_internal", b"param_value_internal", "trial_id", b"trial_id"]) -> None: ...

global___SetTrialParameterRequest = SetTrialParameterRequest

@typing.final
class SetTrialParameterReply(google.protobuf.message.Message):
    """*
    Reply to set a trial parameter.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetTrialParameterReply = SetTrialParameterReply

@typing.final
class GetTrialIdFromStudyIdTrialNumberRequest(google.protobuf.message.Message):
    """*
    Request to get a trial id from its study id and trial number.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    TRIAL_NUMBER_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    trial_number: builtins.int
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        trial_number: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["study_id", b"study_id", "trial_number", b"trial_number"]) -> None: ...

global___GetTrialIdFromStudyIdTrialNumberRequest = GetTrialIdFromStudyIdTrialNumberRequest

@typing.final
class GetTrialIdFromStudyIdTrialNumberReply(google.protobuf.message.Message):
    """*
    Reply to get a trial id from its study id and trial number.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trial_id", b"trial_id"]) -> None: ...

global___GetTrialIdFromStudyIdTrialNumberReply = GetTrialIdFromStudyIdTrialNumberReply

@typing.final
class SetTrialStateValuesRequest(google.protobuf.message.Message):
    """*
    Request to set trial state and values.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    state: global___TrialState.ValueType
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        state: global___TrialState.ValueType = ...,
        values: collections.abc.Iterable[builtins.float] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["state", b"state", "trial_id", b"trial_id", "values", b"values"]) -> None: ...

global___SetTrialStateValuesRequest = SetTrialStateValuesRequest

@typing.final
class SetTrialStateValuesReply(google.protobuf.message.Message):
    """*
    Reply to set trial state and values.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_UPDATED_FIELD_NUMBER: builtins.int
    trial_updated: builtins.bool
    def __init__(
        self,
        *,
        trial_updated: builtins.bool = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trial_updated", b"trial_updated"]) -> None: ...

global___SetTrialStateValuesReply = SetTrialStateValuesReply

@typing.final
class SetTrialIntermediateValueRequest(google.protobuf.message.Message):
    """*
    Request to set a trial intermediate value.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    STEP_FIELD_NUMBER: builtins.int
    INTERMEDIATE_VALUE_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    step: builtins.int
    intermediate_value: builtins.float
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        step: builtins.int = ...,
        intermediate_value: builtins.float = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["intermediate_value", b"intermediate_value", "step", b"step", "trial_id", b"trial_id"]) -> None: ...

global___SetTrialIntermediateValueRequest = SetTrialIntermediateValueRequest

@typing.final
class SetTrialIntermediateValueReply(google.protobuf.message.Message):
    """*
    Reply to set a trial intermediate value.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetTrialIntermediateValueReply = SetTrialIntermediateValueReply

@typing.final
class SetTrialUserAttributeRequest(google.protobuf.message.Message):
    """*
    Request to set a trial user attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    key: builtins.str
    value: builtins.str
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        key: builtins.str = ...,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["key", b"key", "trial_id", b"trial_id", "value", b"value"]) -> None: ...

global___SetTrialUserAttributeRequest = SetTrialUserAttributeRequest

@typing.final
class SetTrialUserAttributeReply(google.protobuf.message.Message):
    """*
    Reply to set a trial user attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetTrialUserAttributeReply = SetTrialUserAttributeReply

@typing.final
class SetTrialSystemAttributeRequest(google.protobuf.message.Message):
    """*
    Request to set a trial system attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    KEY_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    key: builtins.str
    value: builtins.str
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        key: builtins.str = ...,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["key", b"key", "trial_id", b"trial_id", "value", b"value"]) -> None: ...

global___SetTrialSystemAttributeRequest = SetTrialSystemAttributeRequest

@typing.final
class SetTrialSystemAttributeReply(google.protobuf.message.Message):
    """*
    Reply to set a trial system attribute.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    def __init__(
        self,
    ) -> None: ...

global___SetTrialSystemAttributeReply = SetTrialSystemAttributeReply

@typing.final
class GetTrialRequest(google.protobuf.message.Message):
    """*
    Request to get a trial by its ID.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_ID_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trial_id", b"trial_id"]) -> None: ...

global___GetTrialRequest = GetTrialRequest

@typing.final
class GetTrialReply(google.protobuf.message.Message):
    """*
    Reply to get a trial by its ID.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIAL_FIELD_NUMBER: builtins.int
    @property
    def trial(self) -> global___Trial: ...
    def __init__(
        self,
        *,
        trial: global___Trial | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["trial", b"trial"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["trial", b"trial"]) -> None: ...

global___GetTrialReply = GetTrialReply

@typing.final
class GetAllTrialsRequest(google.protobuf.message.Message):
    """*
    Request to get all trials in a study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    STUDY_ID_FIELD_NUMBER: builtins.int
    STATES_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    @property
    def states(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___TrialState.ValueType]: ...
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        states: collections.abc.Iterable[global___TrialState.ValueType] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["states", b"states", "study_id", b"study_id"]) -> None: ...

global___GetAllTrialsRequest = GetAllTrialsRequest

@typing.final
class GetAllTrialsReply(google.protobuf.message.Message):
    """*
    Reply to get all trials in a study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    TRIALS_FIELD_NUMBER: builtins.int
    @property
    def trials(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Trial]: ...
    def __init__(
        self,
        *,
        trials: collections.abc.Iterable[global___Trial] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["trials", b"trials"]) -> None: ...

global___GetAllTrialsReply = GetAllTrialsReply

@typing.final
class Study(google.protobuf.message.Message):
    """*
    Study.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class UserAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class SystemAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    STUDY_ID_FIELD_NUMBER: builtins.int
    STUDY_NAME_FIELD_NUMBER: builtins.int
    DIRECTIONS_FIELD_NUMBER: builtins.int
    USER_ATTRIBUTES_FIELD_NUMBER: builtins.int
    SYSTEM_ATTRIBUTES_FIELD_NUMBER: builtins.int
    study_id: builtins.int
    study_name: builtins.str
    @property
    def directions(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[global___StudyDirection.ValueType]: ...
    @property
    def user_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    @property
    def system_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    def __init__(
        self,
        *,
        study_id: builtins.int = ...,
        study_name: builtins.str = ...,
        directions: collections.abc.Iterable[global___StudyDirection.ValueType] | None = ...,
        user_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        system_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["directions", b"directions", "study_id", b"study_id", "study_name", b"study_name", "system_attributes", b"system_attributes", "user_attributes", b"user_attributes"]) -> None: ...

global___Study = Study

@typing.final
class Trial(google.protobuf.message.Message):
    """*
    Trial.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class ParamsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.float
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.float = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class DistributionsEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class UserAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class SystemAttributesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.str
        value: builtins.str
        def __init__(
            self,
            *,
            key: builtins.str = ...,
            value: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    @typing.final
    class IntermediateValuesEntry(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        KEY_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        key: builtins.int
        value: builtins.float
        def __init__(
            self,
            *,
            key: builtins.int = ...,
            value: builtins.float = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["key", b"key", "value", b"value"]) -> None: ...

    TRIAL_ID_FIELD_NUMBER: builtins.int
    NUMBER_FIELD_NUMBER: builtins.int
    STATE_FIELD_NUMBER: builtins.int
    VALUES_FIELD_NUMBER: builtins.int
    DATETIME_START_FIELD_NUMBER: builtins.int
    DATETIME_COMPLETE_FIELD_NUMBER: builtins.int
    PARAMS_FIELD_NUMBER: builtins.int
    DISTRIBUTIONS_FIELD_NUMBER: builtins.int
    USER_ATTRIBUTES_FIELD_NUMBER: builtins.int
    SYSTEM_ATTRIBUTES_FIELD_NUMBER: builtins.int
    INTERMEDIATE_VALUES_FIELD_NUMBER: builtins.int
    trial_id: builtins.int
    number: builtins.int
    state: global___TrialState.ValueType
    datetime_start: builtins.str
    datetime_complete: builtins.str
    @property
    def values(self) -> google.protobuf.internal.containers.RepeatedScalarFieldContainer[builtins.float]: ...
    @property
    def params(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.float]: ...
    @property
    def distributions(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    @property
    def user_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    @property
    def system_attributes(self) -> google.protobuf.internal.containers.ScalarMap[builtins.str, builtins.str]: ...
    @property
    def intermediate_values(self) -> google.protobuf.internal.containers.ScalarMap[builtins.int, builtins.float]: ...
    def __init__(
        self,
        *,
        trial_id: builtins.int = ...,
        number: builtins.int = ...,
        state: global___TrialState.ValueType = ...,
        values: collections.abc.Iterable[builtins.float] | None = ...,
        datetime_start: builtins.str = ...,
        datetime_complete: builtins.str = ...,
        params: collections.abc.Mapping[builtins.str, builtins.float] | None = ...,
        distributions: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        user_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        system_attributes: collections.abc.Mapping[builtins.str, builtins.str] | None = ...,
        intermediate_values: collections.abc.Mapping[builtins.int, builtins.float] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["datetime_complete", b"datetime_complete", "datetime_start", b"datetime_start", "distributions", b"distributions", "intermediate_values", b"intermediate_values", "number", b"number", "params", b"params", "state", b"state", "system_attributes", b"system_attributes", "trial_id", b"trial_id", "user_attributes", b"user_attributes", "values", b"values"]) -> None: ...

global___Trial = Trial
